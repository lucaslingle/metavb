import collections

DistributionalMemory = collections.namedtuple('DistributionalMemory', field_names=['R', 'U'])
DistributionalAddress = collections.namedtuple('DistributionalAddress', field_names=['Mu_w', 'Sigma_w'])
DistributionalAddresses = collections.namedtuple('DistributionalAddresses', field_names=['Mu_W', 'Sigma_w'])
