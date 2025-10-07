# produits à base de “champagne”
# 10 premiers produits  et les save dans un fichier “.csv”, 
# contenant pour chaque produit les données suivantes : foodId, label, category, foodContentsLabel, image
import requests
import csv

base_url = "https://world.openfoodfacts.org/cgi/search.pl"

params = {
    "action": "process", #  introduces the action to be performed (process)
    "json": 1,
    "page_size": 10,
    "tagtype_0": "categories", #  adds the first search criterion (categories)
    "tag_contains_0": "contains", #determines that the results should be included (note that you can exclude products from the search)
    "tag_0": "champagnes", #  defines the category to be filtered (champagnes)
    "fields": "code,product_name,categories,ingredients_text,image_url"
}

try:
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        products = [
            {
                'foodId': p.get('code', ''),
                'label': p.get('product_name', ''),
                'category': p.get('categories', ''),
                'foodContentsLabel': p.get('ingredients_text', ''),
                'image': p.get('image_url', '')
            }
            for p in data.get('products', [])
        ]
        
        csv_filename = 'produits_champagne.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['foodId', 'label', 'category', 'foodContentsLabel', 'image']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(products)
        
        print(f"✓ {len(products)} produits sauvegardés dans '{csv_filename}'")
    else:
        print(f"Erreur: {response.status_code}")
        
except Exception as e:
    print(f"Erreur: {e}")