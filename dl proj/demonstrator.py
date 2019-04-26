import kivy
import json
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
import os

import sample
from sample import Vocabulary

kivy.require('1.10.1')
Window.size = (1000,600)

presentation = Builder.load_file('demonstrator.kv')

class MainScreen(BoxLayout):

    def selected(self, filename):
        self.ids.img.source = filename[0]
        self.ids.pred.text = sample.run(self.ids.img.source)
        #self.ids.pred.text = 'guess what I have predicted'

class MyApp(App):
    def build(self):
        return MainScreen()

if __name__ == '__main__':
    MyApp().run()