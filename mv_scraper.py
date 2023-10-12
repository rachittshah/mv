import os
import re
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import HtmlResponse
from twisted.internet import reactor
from multiprocessing import Process, Queue


class WebpageSpider(scrapy.Spider):
    name = "webpagespider"

    def __init__(self, *args, **kwargs):
        print("WebpageSpider __init__ called")
        super(WebpageSpider, self).__init__(*args, **kwargs)
        self.start_urls = kwargs.get('start_urls', [])

    def parse(self, response):
        print("parse called")
        # Parse HTML content
        cleaned_text = self.extract_text(response)

        if self.output_file:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(cleaned_text + "\n")

    def extract_text(self, response):
        print("extract_text called")
        # Create a new HtmlResponse without images to avoid downloading them
        cleaned_html = re.sub(r'<img[^>]*>', '', response.text)
        cleaned_html = re.sub(r'<style[^>]*>.*?<\/style>', '', cleaned_html)

        cleaned_response = HtmlResponse(
            url=response.url, body=cleaned_html, encoding='utf-8')

        # Extract text from the cleaned response
        text_content = cleaned_response.xpath('//text()').getall()
        text_content = " ".join(text_content)

        # Remove HTML tags and extra spaces
        cleaned_text = re.sub(r'<[^>]+>', '', text_content)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Change links to text
        cleaned_text = self.replace_links_with_text(cleaned_text)

        return cleaned_text

    def replace_links_with_text(self, text):
        print("replace_links_with_text called")
        # Replace links with text
        replaced_text = re.sub(r'<a\s+.*?>(.*?)<\/a>', r'\1', text)
        return replaced_text


def scrape(urls, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    messages = []
    process = CrawlerProcess()

    def _crawl(queue, spider, *args, **kwargs):
        try:
            print("_crawl called")
            deferred = process.crawl(spider, *args, **kwargs)
            deferred.addBoth(lambda _: reactor.stop())
            reactor.run()
            queue.put(None)
        except Exception as e:
            messages.append(f"Error: {e}")
            queue.put(e)

    queue = Queue()
    crawler_process = Process(target=_crawl, args=(queue, WebpageSpider, 'start_urls', urls, 'output_file', output_file))
    crawler_process.start()
    result = queue.get()
    crawler_process.join()

    if result is not None:
        messages.append(result)
        raise result

    return messages