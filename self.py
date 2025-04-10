# old_model: trained on [3,3,3]
# new_model: built for [3,3,3,2]

old_sd = old_model.policy.state_dict()
new_sd = new_model.policy.state_dict()

# 迁移所有匹配参数（比如 MLP hidden层等）
for k in new_sd:
    if k in old_sd and new_sd[k].shape == old_sd[k].shape:
        new_sd[k] = old_sd[k]

# ✅ 有些参数 shape 不一样（比如 action_net.3.weight），就不迁移

new_model.policy.load_state_dict(new_sd)
