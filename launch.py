import gradio

def hello(inp):
#   return "Hello!"
  return inp

image = gradio.inputs.Image(label="Input image", source="webcam")

# io = gradio.Interface(fn=hello, live=True, inputs=image, outputs='text', title='Hello World', 
#     description='The simplest Hosted interface.')

io = gradio.Interface(fn=hello, live=True, inputs=image, outputs='image', title='Hello World', 
    description='The simplest Hosted interface.')  

io.launch()
