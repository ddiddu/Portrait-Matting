import gradio

def hello(inp):
  return "Hello!"

image = gradio.inputs.Image(label="Input image", source="webcam")

io = gradio.Interface(fn=hello, live=True, inputs=image, outputs='text', title='Hello World', 
    description='The simplest Hosted interface.')  
io.launch()
