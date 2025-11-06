import pygame
import random
from enum import Enum
import CACLA


class PixelArray(pygame.sprite.Sprite):
    height = None
    width = None    
    x_intervals = None
    y_intervals = None
    textpadding = 50
    font = None
    def __init__(self, pos, display,name,event_type) -> None:
        super().__init__(display.pixel_arrays)

        self.display = display
        self.name = name
        self.event_type =event_type
        self.border_values = (0.0,1.0)

        self.image = pygame.Surface((self.width,self.height+self.textpadding))
        self.rect = self.image.get_rect(topleft=pos)
        self.color_array = [[[random.randint(0,255),0,0]for _ in range(self.y_intervals)] for _ in range(self.x_intervals)]
        
        self.fill_surf()
        
    def fill_surf(self):
        self.image.fill(pygame.Color("black"))        
        self.image.blit(self.font.render(f"{self.name}", True, pygame.Color('White')), (5, 5))
        self.image.blit(self.font.render(f"Min: {self.border_values[0]:.2} Max: {self.border_values[1]:.2}", True, pygame.Color('White')), (5, 10+PixelArray.font.size("I")[1]))
        pixel_size_x = self.width/self.x_intervals
        pixel_size_y = self.height/self.y_intervals
        for x in range(self.x_intervals):
            for y in range(self.y_intervals):
                pygame.draw.rect(self.image,self.color_array[x][y],
                    pygame.Rect((x * pixel_size_x, y * pixel_size_y+self.textpadding),
                                (pixel_size_x, pixel_size_y))
                                )

    def update(self,events,dt):
        self.fill_surf()
        for event in events:
            if event.type == self.event_type:
                self.color_array = event.colors
                self.border_values = (event.min,event.max)
                self.fill_surf()
    

            
class Button(pygame.sprite.Sprite):
    width = 200
    height = 40
    font = None
    def __init__(self, pos, color, text, display, action, cacla_state_active_in=None):
        super().__init__(display.buttons)
        self.color = color
        self.action = action
        self.display = display
        self.text = text
        self.cacla_state_active_in = cacla_state_active_in

        self.image = pygame.Surface((self.width, self.height))
        self.rect = self.image.get_rect(topright=pos)
        self.fill_surf(self.color)

    def fill_surf(self, color):
        self.image.fill(pygame.Color(color))
        self.image.blit(self.font.render(self.text, True, pygame.Color('White')), (10, 10))

    def update(self, events, dt):

        # button can only be used if current State of Cacla matches attribute
        if self.cacla_state_active_in != None and self.cacla_state_active_in != CACLA.STATE:
            self.fill_surf('darkgrey')
            return

        self.fill_surf(self.color)
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.rect.collidepoint(event.pos):
                    # if the player clicked the button, the action is invoked
                    self.action()

class InteractivePixelArray(pygame.sprite.Sprite):
    height = None
    width = None
    font = None
    x_size = None
    y_size = None
    actions = []
    def __init__(self, pos, display) -> None:
        super().__init__(display.interactive_pixel_arrays)
        self.display = display
                
        self.image = pygame.Surface((self.width,self.height))
        self.rect = self.image.get_rect(topleft=pos)
        
        self.obstacles_lines = []
        self.pressed_pos = None

        self.fill_surf()
        
    def fill_surf(self):
        self.image.fill(pygame.Color("white"))        
        self.image.blit(self.font.render(f"Interactive Area to draw Obsticales", True, pygame.Color('White')), (self.width/4,5))
        # draw goal
        x,y = self.map_from_CACLA_coordinates(CACLA.World.goal)
        width = self.width * CACLA.World.goal_width/self.x_size
        x = x - width/2
        y = y - width/2
        pygame.draw.rect(self.image,"red",pygame.Rect((x,y),(width,width)))
        # draw obstacles
        for obstacle in self.obstacles_lines:
            pygame.draw.line(self.image,"black",obstacle[0],obstacle[1],width=2)
        # draw actions
        for index in range(len(self.actions)):
            if index + 2 >= len(self.actions):
                break
            else:
                x1,y1 = self.map_from_CACLA_coordinates(self.actions[index])
                x2,y2 = self.map_from_CACLA_coordinates(self.actions[index+1])
                pygame.draw.line(self.image,(0,255/len(self.actions)*index,0),(x1,y1),(x2,y2))
                pygame.draw.circle(self.image,"blue",(x1,y1),radius=3.0)
    
    def map_from_CACLA_coordinates(self,coordinate):
        x,y = coordinate
        return self.width * x/self.x_size, self.height * y/self.y_size
    def map_to_CACLA_coordinates(self,coordinate):
        x,y = coordinate
        return self.x_size * x/self.width, self.y_size * y/self.height
    
    def remove_obstacles(self,index=None):
        if index == None:
            self.obstacles_lines = []
            CACLA.World.line_obstacles = []
        elif index >= len(self.obstacles_lines):
            pass # TODO Do nothing for now
        else:
            del self.obstacles_lines[index]
            del CACLA.World.line_obstacles[index]
        self.fill_surf()

    def update(self,events,dt):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.rect.collidepoint(event.pos):
                    # if the player clicked the button, the action is invoked
                    x = event.pos[0] - self.rect.topleft[0] 
                    y = event.pos[1] - self.rect.topleft[1] 
                    self.pressed_pos = (x,y)
                    pygame.draw.circle(self.image,"grey",center=self.pressed_pos,radius=3)
            if event.type == pygame.MOUSEBUTTONUP and self.pressed_pos != None:
                if self.rect.collidepoint(event.pos):
                    x = (event.pos[0] - self.rect.topleft[0])
                    y = (event.pos[1] - self.rect.topleft[1])
                    self.obstacles_lines.append([self.pressed_pos,(x,y)])
                    # transform from pygame coordinates to pixels
                    x,y = self.map_to_CACLA_coordinates([x,y])
                    self.pressed_pos = self.map_to_CACLA_coordinates(self.pressed_pos)
                    CACLA.World.line_obstacles.append([self.pressed_pos,(x,y)])
                self.fill_surf()
                self.pressed_pos = None
            if event.type == CACLA.EVENT_EVALUATION_ACTIONS:
                self.actions = event.actions
                self.fill_surf()
        if CACLA.STATE == CACLA.States.EVALUATE_ACTIONS and self.rect.collidepoint(pygame.mouse.get_pos()):
           x,y = pygame.mouse.get_pos()
           x -= self.rect.topleft[0]
           y -= self.rect.topleft[1]
           x = x/self.width *self.x_size
           y = y/self.height *self.y_size
           CACLA.World.user_start = (x,y)

class Display:
    def __init__(self):
        PixelArray.x_intervals = CACLA.World.x_features
        PixelArray.y_intervals = CACLA.World.y_features
        InteractivePixelArray.x_size = CACLA.World.size_x
        InteractivePixelArray.y_size = CACLA.World.size_y

        pixel_size = 10
        PixelArray.width = CACLA.World.x_features*pixel_size
        PixelArray.height = CACLA.World.y_features*pixel_size
        InteractivePixelArray.width = PixelArray.width*2
        InteractivePixelArray.height = PixelArray.height*2
        padding = 10

        # Initialization of Display
        pygame.init()
        self.surface = pygame.display.set_mode(
            size=(PixelArray.width * 3 + 5*padding+Button.width, PixelArray.height+padding*2+PixelArray.textpadding+InteractivePixelArray.height+padding*3))
        self.surface_rect = self.surface.get_rect()
        self.clock = pygame.time.Clock()

        # Initialization of PixelArrays
        PixelArray.font = pygame.font.SysFont(None, 20)
        self.pixel_arrays = pygame.sprite.Group()
        x = 0
        y = 0
        PixelArray((x+padding,y+padding),self,"Actor horizontal walk direction",CACLA.EVENT_EVALUATION_ACTOR_X)
        PixelArray((x+PixelArray.width+padding*2,y+padding),self,"Actor vertical walk direction",CACLA.EVENT_EVALUATION_ACTOR_Y)
        PixelArray((x+PixelArray.width*2+padding*3,y+padding),self,"Critic value function",CACLA.EVENT_EVALUATION_CRITIC)

        # Initialization of InteractivePixelArray
        InteractivePixelArray.font = pygame.font.SysFont(None, 26)
        self.interactive_pixel_arrays = pygame.sprite.Group()
        x = 0
        y = 0 + PixelArray.height+PixelArray.textpadding +padding*3
        ipa = InteractivePixelArray((x+2*padding,y),self)

        # Initialization of Buttons
        Button.font = pygame.font.SysFont(None, 26)
        self.buttons = pygame.sprite.Group()
        x = self.surface_rect.topright[0]
        y = self.surface_rect.topright[1]
        Button((x-padding,y+padding), 'brown', 'Remove Obstacles', self, lambda:ipa.remove_obstacles())
        
        Button((x-padding,y+Button.height+padding*2), 'green', 'Show Actions', self, lambda:CACLA.set_state(CACLA.States.EVALUATE_ACTIONS),CACLA.States.TRAIN)
        def set_CACLA_train():
            InteractivePixelArray.actions = []
            CACLA.set_state(CACLA.States.TRAIN)
        Button((x-padding,y+Button.height*2+padding*3), 'green', 'Train Model', self, set_CACLA_train,CACLA.States.EVALUATE_ACTIONS)

    def game_loop(self):
        dt = 0
        while True:
            events = pygame.event.get()
            #UPDATE
            for event in events:
                if event.type == pygame.QUIT:
                    quit()
            self.buttons.update(events,dt)
            self.pixel_arrays.update(events,dt)
            self.interactive_pixel_arrays.update(events,dt)

            #DRAW
            self.surface.fill((0, 0, 0))
            self.buttons.draw(self.surface)
            self.pixel_arrays.draw(self.surface)
            self.interactive_pixel_arrays.draw(self.surface)

            pygame.display.flip()
            dt = self.clock.tick(60)


    # pixel_array specifies which pixel_array to set
    # color_array specifies the values to set the array to
    def set_pixel_array(self, pixel_array: int ,color_array):
        pass

def main():
    display = Display()
    CACLA.STATE =CACLA.States.TRAIN
    while True:
        display.game_loop()

if __name__ == "__main__":
    main()
    
