class Scene(object):

    PLACE_DEPTH = 15.0

    def __init__(self):
        # node queue in scene
        self.node_list = list()

    def add_node(self, node):
        """ add a new node in scene"""
        self.node_list.append(node)

    def render(self):
        """traverse all the nodes in scene and rend them"""
        for node in self.node_list:
            node.render()

