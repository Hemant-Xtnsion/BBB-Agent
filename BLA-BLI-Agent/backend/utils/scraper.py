import httpx
from bs4 import BeautifulSoup
from typing import List, Dict
import json
import asyncio
from pathlib import Path


class BlaBliScraper:
    """Scraper for blabliblulife.com products"""
    
    def __init__(self, base_url: str = "https://blabliblulife.com"):
        self.base_url = base_url
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    async def fetch_page(self, url: str) -> str:
        """Fetch a page content"""
        async with httpx.AsyncClient(headers=self.headers, timeout=30.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                return ""
    
    def parse_product_page(self, html: str, url: str) -> Dict:
        """Parse individual product page"""
        soup = BeautifulSoup(html, 'lxml')
        
        product = {
            "title": "",
            "price": "",
            "thumbnail": "",
            "url": url,
            "description": "",
            "tags": [],
            "category": ""
        }
        
        # Extract product title
        title_elem = soup.find('h1', class_='product-title') or soup.find('h1')
        if title_elem:
            product["title"] = title_elem.get_text(strip=True)
        
        # Extract price
        price_elem = soup.find('span', class_='price') or soup.find(class_='product-price')
        if price_elem:
            product["price"] = price_elem.get_text(strip=True)
        
        # Extract thumbnail/image
        img_elem = soup.find('img', class_='product-image') or soup.find('img')
        if img_elem:
            img_src = img_elem.get('src') or img_elem.get('data-src')
            if img_src:
                if img_src.startswith('//'):
                    product["thumbnail"] = 'https:' + img_src
                elif img_src.startswith('/'):
                    product["thumbnail"] = self.base_url + img_src
                else:
                    product["thumbnail"] = img_src
        
        # Extract description
        desc_elem = soup.find('div', class_='product-description') or soup.find('div', class_='description')
        if desc_elem:
            product["description"] = desc_elem.get_text(strip=True)[:500]  # Limit to 500 chars
        
        # Extract tags/categories
        tags = []
        tag_elems = soup.find_all('a', class_='tag') or soup.find_all(class_='product-tag')
        for tag in tag_elems:
            tags.append(tag.get_text(strip=True))
        product["tags"] = tags
        
        # Extract category
        category_elem = soup.find('span', class_='category') or soup.find(class_='product-category')
        if category_elem:
            product["category"] = category_elem.get_text(strip=True)
        
        return product
    
    async def get_product_links(self) -> List[str]:
        """Get all product links from the website"""
        product_links = []
        
        # Try common product listing pages
        pages_to_check = [
            f"{self.base_url}/shop",
            f"{self.base_url}/products",
            f"{self.base_url}/collections/all",
            self.base_url
        ]
        
        for page_url in pages_to_check:
            html = await self.fetch_page(page_url)
            if not html:
                continue
                
            soup = BeautifulSoup(html, 'lxml')
            
            # Find product links
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                
                # Filter for product URLs
                if any(keyword in href.lower() for keyword in ['/product', '/item', '/p/']):
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    elif href.startswith('http'):
                        full_url = href
                    else:
                        continue
                    
                    if full_url not in product_links:
                        product_links.append(full_url)
        
        return product_links
    
    async def scrape_all_products(self) -> List[Dict]:
        """Scrape all products from the website"""
        print("[*] Fetching product links...")
        product_links = await self.get_product_links()
        print(f"[+] Found {len(product_links)} product links")
        
        products = []
        
        # Limit to first 50 products to avoid overwhelming
        for i, link in enumerate(product_links[:50], 1):
            print(f"[*] Scraping product {i}/{min(len(product_links), 50)}: {link}")
            html = await self.fetch_page(link)
            if html:
                product = self.parse_product_page(html, link)
                if product["title"]:  # Only add if we got a title
                    products.append(product)
            
            # Be nice to the server
            await asyncio.sleep(0.5)
        
        return products
    
    async def save_products(self, output_path: str = "data/products.json"):
        """Scrape and save products to JSON file"""
        products = await self.scrape_all_products()
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(products, f, indent=2, ensure_ascii=False)
        
        print(f"[+] Saved {len(products)} products to {output_path}")
        return products


async def main():
    """Main function to run the scraper"""
    scraper = BlaBliScraper()
    await scraper.save_products()


if __name__ == "__main__":
    asyncio.run(main())
