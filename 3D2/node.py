import random
from OpenGL.GL import glCallList, glColor3f, glMaterialfv, glMultMatrixf, glPopMatrix, glPushMatrix, \
                              GL_EMISSION, GL_FRONT
import numpy
import color
from primtive import G_OBJ_CUBE, G_OBJ_SPHERE
from transformation import scaling, translation
from aabb import AABB

class Node(object):
    def __init__(self):
        # the color number of this node
        self.color_index = random.randint(color.MIN_COLOR, color.MAX_COLOR)

        self.aabb = AABB([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        # the pingyi matrix of this node, determines the position of the node in scene
        self.translation_matrix = numpy.identity(4)
        # the suofang matrix of this node, determines the size of the node
        self.scaling_matrix = numpy.identity(4)
        self.selected = False


    def render(self):
        glPushMatrix()

        # pingyi shixian
        glMultMatrixf(numpy.transpose(self.translation_matrix))
        # suofang shixian
        glMultMatrixf(self.scaling_matrix)
        cur_color = color.COLORS[self.color_index]
        #set color
        glColor3f(cur_color[0], cur_color[1], cur_color[2])
        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.3, 0.3, 0.3])

        self.render_self()
        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0])
        glPopMatrix()

    def select(self, select = None):
        if select is not None:
            self.selected = select
        else:
            self.selected = not self.selected

    def render_self(self):
        raise NotImplementedError(
                "the abstract node class doesn't define 'render_self'")

    def translate(self, x, y, z):
        self.translation_matrix = numpy.dot(self.translation_matrix, translation([x, y, z]))

    def scale(self, s):
        self.scaling_matrix = numpy.dot(self.scaling_matrix, scaling([s, s, s]))

class Primitive(Node):
    def __init__(self):
        super(Primitive, self).__init__()
        self.call_list = None

    def render_self(self):
        glCallList(self.call_list)


class Sphere(Primitive):
    """ball"""
    def __init__(self):
        super(Sphere, self).__init__()
        self.call_list = G_OBJ_SPHERE
    def pick(self, start, direction, mat):
        newmat = numpy.dot(
             numpy.dot(mat, self.translation_matrix),
             numpy.linalg.inv(self.scaling_matrix)
        )
        results = self.aabb.ray_hit(start, direction, newmat)
        return results


class Cube(Primitive):
    def __init__(self):
        super(Cube, self).__init__()
        self.call_list = G_OBJ_CUBE
    def pick(self, start, direction, mat):
        newmat = numpy.dot(
            numpy.dot(mat, self.translation_matrix),
            numpy.linalg.inv(self.scaling_matrix)
        )
        results = self.aabb.ray_hit(start, direction, newmat)
        return results

class HierarchicalNode(Node):
    def __init__(self):
        super(HierarchicalNode, self).__init__()
        self.child_nodes = []

    def render_self(self):
        for child in self.child_nodes:
            child.render()


class SnowFigure(HierarchicalNode):
    def __init__(self):
        super(SnowFigure, self).__init__()
        self.child_nodes = [Sphere(), Sphere(), Sphere()]
        self.child_nodes[0].translate(0, -0.6, 0)
        self.child_nodes[1].translate(0, 0.1, 0)
        self.child_nodes[1].scale(0.8)
        self.child_nodes[2].translate(0, 0.75, 0)
        self.child_nodes[2].scale(0.7)
        for child_node  in self.child_nodes:
            child_node.color_index = color.MIN_COLOR


    def pick(self, start, direction, mat):
        newmat = numpy.dot(
            numpy.dot(mat, self.translation_matrix),
            numpy.linalg.inv(self.scaling_matrix)
        )
        result = self.aabb.ray_hit(start, direction, newmat)
        return result
