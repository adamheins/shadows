

class Text:
    """Text label."""

    def __init__(self, text, font, position, color):
        self.text = text
        self.font = font
        self.color = color
        self.position = position

        self.update()

    def update(self, text=None, position=None, color=None):
        """Update the text label."""
        if text is not None:
            self.text = text
        if color is not None:
            self.color = color
        if position is not None:
            self.position = position
        self.image = self.font.render(self.text, True, self.color)

    @property
    def shape(self):
        return (self.image.get_width(), self.image.get_height())

    @property
    def rect(self):
        return self.image.get_rect()

    def draw(self, surface):
        """Draw the text on the surface."""
        surface.blit(self.image, self.position)


