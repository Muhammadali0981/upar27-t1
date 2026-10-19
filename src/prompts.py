"""Natural-language prompts per attribute (N1). POS: attribute present, NEG: absent."""

_UB = "upper-body clothing"
_LB = "lower-body clothing"


def _ub_color(c):
    return [f"a surveillance photo of a person wearing a {c} top",
            f"a photo of a pedestrian in a {c} shirt or jacket",
            f"a person whose {_UB} is {c}",
            f"a low-resolution photo of a person wearing {c} clothes on the upper body"]


def _lb_color(c):
    return [f"a surveillance photo of a person wearing {c} trousers",
            f"a photo of a pedestrian in {c} pants or a {c} skirt",
            f"a person whose {_LB} is {c}",
            f"a low-resolution photo of a person wearing {c} clothes on the lower body"]


_COLOR_WORD = {"Other": "multicoloured", "Grey": "grey"}

POS = {
    "Age-Young": ["a surveillance photo of a child", "a photo of a young kid walking", "a photo of a teenager",
                  "a low-resolution photo of a small child"],
    "Age-Adult": ["a surveillance photo of an adult", "a photo of a grown-up person walking",
                  "a photo of a middle-aged adult", "a low-resolution photo of an adult pedestrian"],
    "Age-Old": ["a surveillance photo of an elderly person", "a photo of an old person walking",
                "a photo of a senior citizen with grey hair", "a low-resolution photo of an old man or woman"],
    "Gender-Female": ["a surveillance photo of a woman", "a photo of a female pedestrian",
                      "a photo of a girl or woman walking", "a low-resolution photo of a woman"],
    "Hair-Length-Short": ["a photo of a person with short hair", "a surveillance photo of a pedestrian with short hair",
                          "a person with a short haircut"],
    "Hair-Length-Long": ["a photo of a person with long hair", "a surveillance photo of a pedestrian with long hair",
                         "a person with shoulder-length or longer hair", "a person with a ponytail"],
    "Hair-Length-Bald": ["a photo of a bald person", "a surveillance photo of a bald pedestrian",
                         "a person with a shaved head"],
    "UpperBody-Length-Short": ["a photo of a person wearing a short-sleeved shirt",
                               "a surveillance photo of a pedestrian in a t-shirt",
                               "a person with bare arms wearing short sleeves"],
    "LowerBody-Length-Short": ["a photo of a person wearing shorts", "a surveillance photo of a pedestrian in a short skirt",
                               "a person with bare legs wearing shorts or a short skirt"],
    "LowerBody-Type-Trousers&Shorts": ["a photo of a person wearing trousers", "a surveillance photo of a pedestrian in pants",
                                       "a person wearing jeans or shorts"],
    "LowerBody-Type-Skirt&Dress": ["a photo of a person wearing a skirt", "a surveillance photo of a woman in a dress",
                                   "a person wearing a skirt or a dress"],
    "Accessory-Backpack": ["a surveillance photo of a person carrying a backpack", "a photo of a pedestrian with a backpack",
                           "a person wearing a rucksack on the back"],
    "Accessory-Bag": ["a surveillance photo of a person carrying a bag", "a photo of a pedestrian with a handbag",
                      "a person carrying a shoulder bag or a shopping bag"],
    "Accessory-Glasses-Normal": ["a photo of a person wearing glasses", "a surveillance photo of a pedestrian with eyeglasses",
                                 "a person wearing spectacles"],
    "Accessory-Glasses-Sun": ["a photo of a person wearing sunglasses", "a surveillance photo of a pedestrian with dark sunglasses",
                              "a person wearing shades"],
    "Accessory-Hat": ["a photo of a person wearing a hat", "a surveillance photo of a pedestrian wearing a cap",
                      "a person with a hat on the head"],
}

NEG = {
    "Age-Young": ["a surveillance photo of an adult"],
    "Age-Adult": ["a surveillance photo of a child or an elderly person"],
    "Age-Old": ["a surveillance photo of a young person"],
    "Gender-Female": ["a surveillance photo of a man", "a photo of a male pedestrian"],
    "Hair-Length-Short": ["a photo of a person with long hair or no hair"],
    "Hair-Length-Long": ["a photo of a person with short hair"],
    "Hair-Length-Bald": ["a photo of a person with hair"],
    "UpperBody-Length-Short": ["a photo of a person wearing a long-sleeved shirt or jacket"],
    "LowerBody-Length-Short": ["a photo of a person wearing long trousers"],
    "LowerBody-Type-Trousers&Shorts": ["a photo of a person wearing a skirt or a dress"],
    "LowerBody-Type-Skirt&Dress": ["a photo of a person wearing trousers"],
    "Accessory-Backpack": ["a photo of a person without a backpack"],
    "Accessory-Bag": ["a photo of a person carrying nothing in the hands"],
    "Accessory-Glasses-Normal": ["a photo of a person without glasses"],
    "Accessory-Glasses-Sun": ["a photo of a person without sunglasses"],
    "Accessory-Hat": ["a photo of a person with no hat"],
}

for _c in ["Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink", "Purple", "Red", "White", "Yellow", "Other"]:
    w = _COLOR_WORD.get(_c, _c.lower())
    POS[f"UpperBody-Color-{_c}"] = _ub_color(w)
    POS[f"LowerBody-Color-{_c}"] = _lb_color(w)
    NEG[f"UpperBody-Color-{_c}"] = ["a photo of a person wearing a top of another colour"]
    NEG[f"LowerBody-Color-{_c}"] = ["a photo of a person wearing trousers of another colour"]
