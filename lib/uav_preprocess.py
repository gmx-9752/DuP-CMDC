from PIL import Image, ImageOps


def preprocess_uav_pil(
    img_pil,
    mode="resize",
    target_width=1024,
    target_height=576,
    pad_multiple=64,
):
    img_pil = img_pil.convert("RGB")
    orig_width, orig_height = img_pil.size

    meta = {
        "orig_width": orig_width,
        "orig_height": orig_height,
        "mode": mode,
        "target_width": target_width,
        "target_height": target_height,
        "pad_multiple": pad_multiple,
        "pad_left": 0,
        "pad_top": 0,
        "pad_right": 0,
        "pad_bottom": 0,
    }

    if mode == "resize":
        img_pil = img_pil.resize((target_width, target_height), Image.BICUBIC)
        meta["proc_width"], meta["proc_height"] = img_pil.size
        return img_pil, meta

    if mode == "none":
        meta["proc_width"], meta["proc_height"] = img_pil.size
        return img_pil, meta

    if mode == "pad64":
        width, height = img_pil.size

        new_width = ((width + pad_multiple - 1) // pad_multiple) * pad_multiple
        new_height = ((height + pad_multiple - 1) // pad_multiple) * pad_multiple

        pad_right = new_width - width
        pad_bottom = new_height - height

        img_pil = ImageOps.expand(
            img_pil,
            border=(0, 0, pad_right, pad_bottom),
            fill=(0, 0, 0),
        )

        meta["pad_right"] = pad_right
        meta["pad_bottom"] = pad_bottom
        meta["proc_width"], meta["proc_height"] = img_pil.size
        return img_pil, meta

    raise ValueError(f"Unknown preprocess mode: {mode}")