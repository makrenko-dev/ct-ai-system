def split_into_tiles(img, tile_size=850, overlap=0.4):
    h, w = img.shape[:2]
    step = int(tile_size * (1 - overlap))

    tiles = []
    for y in range(0, max(h - tile_size + 1, 1), step):
        for x in range(0, max(w - tile_size + 1, 1), step):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)

            tile = img[y:y2, x:x2]
            if tile.shape[0] < 50 or tile.shape[1] < 50:
                continue

            tiles.append({
                "img": tile,
                "offset": (x, y)
            })
    return tiles
