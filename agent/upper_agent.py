
class UpperAgent:
    def __init__(self):
        self.current_state = 0
        self.states = {0: 'attack', 1: 'guide', 2: 'dodge'}

    def reset(self):
        self.current_state = 0

    def select_maneuver_model(self, self_missile_working, opponent_missile_working):
        if opponent_missile_working:
            # 只要敌方有弹在攻击自己，就在防御状态
            self.current_state = 2
        else:
            # 没弹了就考虑进攻
            if self_missile_working:
                self.current_state = 1
            else:
                self.current_state = 0
        return self.current_state
