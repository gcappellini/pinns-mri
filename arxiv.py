import urllib.parse, urllib.request
import xml.etree.ElementTree as ET
import csv

q = (
  "(ti:(physics-informed neural networks) OR abs:(physics-informed neural networks)) "
  "AND (ti:(magnetic resonance imaging) OR ti:(MRI) OR abs:(magnetic resonance imaging) OR abs:(MRI)) "
  "AND submittedDate:[202001010000 TO 202512312359]"
)
params = {
    'search_query': q,
    'start': 0,
    'max_results': 200,

}
url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)

req = urllib.request.Request(
    url,
    headers={"User-Agent": "MyArxivClient/1.0 (email@example.com)"}
)
with urllib.request.urlopen(req) as resp:
    xml = resp.read().decode('utf-8')

root = ET.fromstring(xml)
ns   = {'atom': 'http://www.w3.org/2005/Atom'}
entries = root.findall('atom:entry', ns)

# Write out to CSV
with open('arXiv.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # header row
    writer.writerow(['id', 'title', 'abstract', 'published', 'link'])
    for e in entries:
        eid       = e.find('atom:id', ns).text
        title     = e.find('atom:title', ns).text.strip().replace('\n', ' ')
        abstract     = e.find('atom:summary', ns).text.strip().replace('\n', ' ')
        published = e.find('atom:published', ns).text
        link      = e.find("atom:link[@type='text/html']", ns).attrib['href']
        writer.writerow([eid, title, abstract, published, link])

print(f"Saved {len(entries)} records to arXiv.csv")