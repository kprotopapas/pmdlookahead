from environments.gridworlds import GridworldEnv

gw = GridworldEnv()

while True:
    a = int(input())
    state, action, info = gw.step(a)
    print(gw.state_to_pos(state), action)
    if info['terminal']:
        print('terminated')
