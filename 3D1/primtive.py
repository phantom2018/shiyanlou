from OpenGL.GL import glBegin, glColor3f, glEnd, glEndList, glLineWidth, glNewList, glNormal3f, glVertex3f, \
                              GL_COMPILE, GL_LINES, GL_QUADS
from OpenGL.GLU import gluDeleteQuadric, gluNewQuadric, gluSphere

G_OBJ_SPHERE = 2
G_OBJ_PLANE = 1
G_OBJ_CUBE = 3

def make_sphere():
    """create ball's rend function list"""
    glNewList(G_OBJ_SPHERE, GL_COMPILE)
    quad = gluNewQuadric()
    gluSphere(quad, 0.5, 30, 30)
    gluDeleteQuadric(quad)
    glEndList()

def init_primitives():
    """init all tuyuan's rend function list"""
    make_sphere()

def make_plane():
    glNewList(G_OBJ_PLANE, GL_COMPILE)
    glBegin(GL_LINES)
    glColor3f(0, 0, 0)
    for i in range(41):
        glVertex3f(-10.0 + 0.5* i, 0, -10)
        glVertex3f(-10.0 + 0.5* i, 0, 10)
        glVertex3f(-10.0, 0, -10+0.5*i)
        glVertex3f(10.0, 0, -10+0.5*i)

    #Axes
    glEnd()
    glLineWidth(5)

    glBegin(GL_LINES)
    glColor3f(0.5, 0.7, 0.5)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(5, 0.0, 0.0)
    glEnd()

    glBegin(GL_LINES)
    glColor3f(0.5, 0.7, 0.5)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 5, 0.0)

    glBegin(GL_LINES)
    glColor3f(0.5, 0.7, 0.5)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 5)

    #y

