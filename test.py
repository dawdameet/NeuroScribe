import yaml
with open('test.yaml','r') as read_file :
    conenets=yaml.safe_load(read_file)
    print(conenets)