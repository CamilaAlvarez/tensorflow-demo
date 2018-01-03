from network.vgg16 import VGG16


class NetworkFactory:
    networks = {'vgg16': VGG16}

    @classmethod
    def get_network(cls, network_name):
        try:
            return cls.networks[network_name]
        except KeyError:
            raise RuntimeError('Invalid network: {}'.format(network_name))