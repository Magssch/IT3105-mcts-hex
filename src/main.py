from world.ledge import Ledge

if __name__ == "__main__":
    world = Ledge()
    for i in range(15):
        print(world.index_to_tuple(i))
