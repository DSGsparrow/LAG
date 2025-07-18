def threat_distance(distance):
    if distance > 10000:
        return 0  # 无威胁
    elif 8000 < distance <= 10000:
        return 1  # 中距（低威胁，特殊处理）
    elif 6000 < distance <= 8000:
        return 2  # 中距（低威胁，特殊处理）
    else:
        return 3  # 高威胁（近距）

def threat_angle(angle):
    if angle > 50:
        return 0
    elif 20 < angle <= 50:
        return 1
    elif 10 < angle <= 20:
        return 2
    else:
        return 3

def threat_speed(speed):
    if speed < 200:
        return 1
    elif 200 <= speed <= 340:
        return 2
    else:
        return 3

def threat_alt_diff(alt_diff):
    if alt_diff < 0:
        return 0
    elif 0 <= alt_diff <= 2000:
        return 3
    else:
        return 2

def fuzzy_should_attack(distance, angle, speed, alt_diff):
    dist_threat = threat_distance(distance)
    angle_threat = threat_angle(angle)
    speed_threat = threat_speed(speed)
    alt_threat = threat_alt_diff(alt_diff)

    # 所有威胁等级
    levels = {
        'distance': dist_threat,
        'angle': angle_threat,
        'speed': speed_threat,
        'altitude_diff': alt_threat
    }

    # 情况 1：所有因素都 >= 中等威胁（等级 >= 2）
    if all(level >= 2 for level in levels.values()):
        return 1

    # 情况 2：距离是中距（等级 = 1），其他三个都是高威胁（等级 = 3）
    if dist_threat == 1 and angle_threat == 3 and speed_threat == 3 and alt_threat == 3:
        return 1

    return 0  # 其他情况不打
