from app import predict
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.logger import Logger, LOG_LEVELS
Logger.setLevel(LOG_LEVELS["debug"])
import os


class LoadSourceDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class LoadDestDialog(FloatLayout):
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    source_player = ObjectProperty(None)
    dest_player = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load_source(self):
        content = LoadSourceDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_load_dest(self):
        content = LoadDestDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename, id):
        print(path, filename, id)
        if id=='source': 
            op_path = predict(os.path.join(path,filename[0]))
            print(op_path)
            self.source_player.source = op_path
        else: 
            self.dest_player.source = os.path.join(path,filename[0])
            print('Dest Source Set')
        self.dismiss_popup()
        


class Editor(App):
    pass


Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadSourceDialog)
Factory.register('SaveDialog', cls=LoadDestDialog)


if __name__ == '__main__':
    Editor().run()
