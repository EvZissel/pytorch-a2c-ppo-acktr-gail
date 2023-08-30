from PIL import Image, ImageFont, ImageDraw
import cv2
import os

font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
font = ImageFont.truetype(font_path, size=16)
# font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 25)
provided_font = ImageFont.load_default()
provided_font = ImageFont.truetype("ariel.ttf", 16)
img = Image.new("RGBA", (200,200), (120,20,20))
draw = ImageDraw.Draw(img)
draw.text((0,0), "This is a test", (255,255,0), font=font)
img.save("a_test.png")