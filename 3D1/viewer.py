from OpenGL.GL import glCallList, glClear, glClearColor, glColorMaterial, glCullFace, glDepthFunc, glDisable, glEnable,\
                      glFlush, glGetFloatv, glLightfv, glLoadIdentity, glMatrixMode, glMultMatrixf, glPopMatrix, \
                      glPushMatrix, glTranslated, glViewport, \
                      GL_AMBIENT_AND_DIFFUSE, GL_BACK, GL_CULL_FACE, GL_COLOR_BUFFER_BIT, GL_COLOR_MATERIAL, \
                      GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_FRONT_AND_BACK, GL_LESS, GL_LIGHT0, GL_LIGHTING, \
                      GL_MODELVIEW, GL_MODELVIEW_MATRIX, GL_POSITION, GL_PROJECTION, GL_SPOT_DIRECTION
from OpenGL.constants import GLfloat_3, GLfloat_4
from OpenGL.GLU import gluPerspective, gluUnProject
from OpenGL.GLUT import glutCreateWindow, glutDisplayFunc, glutGet, glutInit, glutInitDisplayMode, \
                        glutInitWindowSize, glutMainLoop, \
                        GLUT_SINGLE, GLUT_RGB, GLUT_WINDOW_HEIGHT, GLUT_WINDOW_WIDTH, glutCloseFunc
import numpy
from numpy.linalg import norm, inv
import random
from OpenGL.GL import glBegin, glColor3f, glEnd, glEndList, glLineWidth, glNewList, glNormal3f, glVertex3f, \
                      GL_COMPILE, GL_LINES, GL_QUADS
from OpenGL.GLU import gluDeleteQuadric, gluNewQuadric, gluSphere

import color
from scene import Scene
from primtive import init_primitives , G_OBJ_PLANE
from node import Sphere, SnowFigure
from node import SnowFigure
from interaction import Interaction

class Viewer(object):
    def __init__(self):
        """ Initialize the viewer. """

        self.init_interface()

        self.init_opengl()
        self.init_scene()

        self.init_interaction()
        init_primitives()
        init_primitives()

    def init_interface(self):

        glutInit()
        glutInitWindowSize(640, 480)
        glutCreateWindow("3D Modeller")
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        #register render function
        glutDisplayFunc(self.render)

    def init_opengl(self):

        #model view matrix
        self.inverseModelView = numpy.identity(4)
        #its anti-matrix
        self.modelView = numpy.identity(4)

        #open tichu effect
        glEnable(GL_CULL_FACE)
        #to not rend invisible part
        glCullFace(GL_BACK)
        #open depth test
        glEnable(GL_DEPTH_TEST)
        #objects being covered ot to rend
        glDepthFunc(GL_LESS)
        #open light 0
        glEnable(GL_LIGHT0)
        #to set the position of light
        glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(0, 0, 1, 0))
        #to set the direction that light sheds at
        glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, GLfloat_3(0, 0, -1))
        #to set material color
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_COLOR_MATERIAL)
        #set the color of a clear-screen
        glClearColor(0.4, 0.4, 0.4, 0.0)

    def init_scene(self):

        self.scene = Scene()

        #to init objects in the scene
        self.create_sample_scene()




    def init_interaction(self):
        self.interaction = Interaction()
        self.interaction.register_callback('pick', self.pick)
        self.interaction.register_callback('move', self.move)
        self.interaction.register_callback('place', self.place)
        self.interaction.register_callback('rotate_color', self.rotate_color)
        self.interaction.register_callback('scale', self.scale)

    def pick(self, x, y):
        #mouse choose a point
        pass

    def move(self, x, y):
        #move the chosen point
        pass

    def place(self, shape, x, y):
        """ put a new node at the mouse's location"""
        pass

    def rotate_color(self, forward):
        """change the color of chosen node"""
        pass

    def scale(self, up):
        """change the size of the chosen node"""
        pass





    def main_loop(self):
        glutMainLoop()

    def render(self):
        #init shadow matrix
        self.init_view()

        #open light
        glEnable(GL_LIGHTING)

        #clear color and depth caches
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #set model view matrix(danwei matrix is ok)
        #set trackball's rotate matrix as Modelview
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        #replace now-matrix with hengdeng matrix
        glLoadIdentity()
        glMultMatrixf(self.interaction.trackball.matrix)

        #save Modelview matrix, later change system with anti-matrix
        currentModelView = numpy.array(glGetFloatv(GL_MODELVIEW_MATRIX))
        self.modelView = numpy.transpose(currentModelView)
        self.inverseModelView = inv(numpy.transpose(currentModelView))


        #rend the scene
        self.scene.render()

        #after each shadow , huifu light state
        glDisable(GL_LIGHTING)
        glCallList(G_OBJ_PLANE)
        glPopMatrix()

        glFlush()


    def init_view(self):
        """init shadow matrix"""
        xSize, ySize = glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT)
        #get screen's kuangaobi
        aspect_ratio = float(xSize) / float(ySize)

        #set shadow matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        #set viewport, should be chonghe with window
        glViewport(0, 0, xSize, ySize)

        #set toushi, eyesight width 70 degree, 1000 distance fron camare
        gluPerspective(70, aspect_ratio, 0.1, 1000.0)

        #jinftou back 15 distances from o
        glTranslated(0, 0, -15)




    def init_scene(self):
        self.scene = Scene()
        self.create_sample_scene()

    def create_sample_scene(self):
        # to create a ball
        sphere_node = Sphere()
        # to set the ball's color
        sphere_node.color_index = 2
        sphere_node.translate(2, 2, 0)
        sphere_node.scale(4)

        #to put the ball in the scene.Default middle of view
        self.scene.add_node(sphere_node)

        #to add a snowman
        hierarchical_node = SnowFigure()
        hierarchical_node.translate(-2, 0, 2)
        hierarchical_node.scale(2)
        self.scene.add_node(hierarchical_node)







if __name__ == "__main__":
    viewer = Viewer()
    viewer.main_loop()

