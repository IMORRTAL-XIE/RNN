class Screen:
    def __init__(self):
        self._width = 1.024
        self._height = 0.768

    @property
    def width(self):
        return self._width
    @width.setter
    def width(self,w):
        self._width = w
    @property
    def height(self):
        return self._height
    @height.setter
    def height(self,h):
        self._height = h
    @property
    def resolution(self):
        return self.width*self._height


s = Screen()
s.width = 1024
s.height = 768
print('resolution =', s.resolution)
if s.resolution == 786432:
    print('测试通过!')
else:
    print('测试失败!')