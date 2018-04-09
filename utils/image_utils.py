def get_rep_from_url(url):

    import urllib.request
    from PIL import Image as PIL_Image
    import shutil
    import requests

    response = requests.get(url, stream=True)
    with open('charpragcap/resources/img.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response