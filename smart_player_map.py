from gym_combat.gym_combat.envs.Common.constants import *
import pickle
import os

# possible_locs_path = 'gym_combat/gym_combat/envs/Greedy/possible_locs_100x100_Berlin.pkl'
# covers_map_path = 'gym_combat/gym_combat/envs/Greedy/covers_map_100x100_Berlin.pkl'
possible_locs_path = 'gym_combat/gym_combat/envs/Greedy/possible_locs_Baqa_Thicken.pkl'
covers_map_path = 'gym_combat/gym_combat/envs/Greedy/covers_map_Baqa_Thicken.pkl'

def calc_possible_locs(enemy_loc, depth=10, neighborhood=8):
    queue = []
    queue.append(enemy_loc)
    res = (100 * DSM).astype(np.uint8)
    res[enemy_loc[0], enemy_loc[1]] = 2
    neighbors = np.asarray([[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]])
    if neighborhood == 4:
        neighbors = np.asarray([[0, -1], [0, 1], [-1, 0], [1, 0]])
    while queue:
        curr = queue.pop(0)
        for nei_dir in neighbors:
            nei = curr + nei_dir
            if nei[0] < 0 or nei[0] >= res.shape[0] or nei[1] < 0 or nei[1] >= res.shape[1]:
                continue
            if res[nei[0], nei[1]]:
                continue
            if res[curr[0], curr[1]] < depth:
                res[nei[0], nei[1]] = res[curr[0], curr[1]] + 1
                queue.append(nei)
    res[res == 0] = depth + 1
    return res

def prepare_possible_locs():
    locs_maps = np.zeros((DSM.shape[0], DSM.shape[1], DSM.shape[0], DSM.shape[1])).astype(np.uint8)
    for enemy_h in range(DSM.shape[0]):
        for enemy_w in range(DSM.shape[1]):
            if DSM[enemy_h, enemy_w]:
                continue
            print(enemy_h, enemy_w)
            enemy_loc = (enemy_h, enemy_w)
            locs_maps[enemy_h, enemy_w, :,:] = calc_possible_locs(enemy_loc)
    f = open(possible_locs_path, 'wb')
    pickle.dump(locs_maps, f)
    return locs_maps

def calc_covers_map(my_map, enemy, max_range):
    enemy = np.asarray(enemy)
    fire_map = np.zeros_like(my_map)
    for j in [0, my_map.shape[1]]:
        for i in range(my_map.shape[0]):
            mark_beyond_clear_range(my_map, np.asarray([i, j]), enemy, fire_map, max_range)
    for i in [0, my_map.shape[0]]:
        for j in range(my_map.shape[1]):
            mark_beyond_clear_range(my_map, np.asarray([i, j]), enemy, fire_map, max_range)
    return fire_map



def mark_beyond_clear_range(obs_map, to_loc, from_loc, fire_map, range = 10000):
    dist = np.linalg.norm(to_loc - from_loc)
    step = 1./dist
    met_wall = False
    for α in np.arange(0, 1, step):
        loc = np.abs((1-α)*from_loc + α*to_loc).astype(int)
        if met_wall and not obs_map[loc[0], loc[1]]:
            met_wall = False
            fire_map[loc[0], loc[1]] = 1
        if obs_map[loc[0], loc[1]]:
            met_wall = True
        if np.linalg.norm(loc - from_loc) > range:
            break

def is_clear_range(reach_time_loc, cover, obs_map):
    ε = 0.00001
    dist = np.linalg.norm(reach_time_loc - cover)
    step = 1./(dist + ε)
    for α in np.arange(0, 1, step):
        loc = np.abs((1-α)*reach_time_loc + α*cover).astype(int)
        if obs_map[loc[0], loc[1]]:
            return False
    return True


def update_killing_range(covers_map, possible_locs, killing_range):
    # for all covers, calc when they will be in kiling range of a possible enemy loc
    covers = np.where(covers_map)
    n_covers = len(covers[0])
    obs_map = possible_locs == 100
    for cover_i in range(n_covers):
        cover = np.asarray([covers[0][cover_i], covers[1][cover_i]])
        final_reach_time = int(possible_locs[cover[1], cover[0]])
        reach_time = 0
        for reach_time in range(2, final_reach_time):
            reach_time_locs = np.where(possible_locs == reach_time)
            n_reach_time_locs = len(reach_time_locs[0])
            found = False
            for reach_time_loc_i in range(n_reach_time_locs):
                reach_time_loc = np.asarray([reach_time_locs[0][reach_time_loc_i], reach_time_locs[1][reach_time_loc_i]])
                if is_clear_range(reach_time_loc, cover, obs_map):
                    found = True
                    break
            if found:
                break
        covers_map[cover[0], cover[1]] = reach_time
    return covers_map

def prepare_covers():
    if os.path.exists(possible_locs_path):
        with open(possible_locs_path, 'rb') as f:
            possible_locs_maps = pickle.load(f)
            print ("found and loaded possible locs")
    else:
        print ("could not found possible_locs. preparing it")
        possible_locs_maps = prepare_possible_locs()

    maps_map = np.zeros((DSM.shape[0], DSM.shape[1], DSM.shape[0], DSM.shape[1])).astype(np.uint8)
    for enemy_h in range(DSM.shape[0]):
        for enemy_w in range(DSM.shape[1]):
            if DSM[enemy_h, enemy_w]:
                continue
            enemy_loc = (enemy_h, enemy_w)
            print (enemy_loc)
            covers_map = calc_covers_map(my_map=DSM, enemy=enemy_loc, max_range=10)
            possible_locs = possible_locs_maps[enemy_loc]
            covers_map = update_killing_range(covers_map=covers_map, possible_locs=possible_locs, killing_range=FIRE_RANGE)
            maps_map[enemy_h, enemy_w, :, :] = covers_map

    f = open(covers_map_path, 'wb')
    pickle.dump(maps_map, f)

if __name__ == '__main__':
    prepare_possible_locs()
    prepare_covers()



