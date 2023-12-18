import numpy as np
from tensor_enum import *
from lego_env import *


#行列出力
#print(Check_Matrix(STABILIZERS).mat)

#LegoEnvからいろいろ出力(うまくいかない)
#print(Biased_Legoenv(max_tensors=14).legs_to_tensor)
#print(Biased_Legoenv(max_tensors=14).render())


#print(T6_Stabilizer([2,4,6]).check_matrix.mat)

#print(T6_Stabilizer([1,2,3,4,5,6]).combine(T6_Stabilizer([7,8,9,10,11,12]),2,4).check_matrix.mat)
#print(T6_Stabilizer([1,2,3,4,5,6]).combine(T6_Stabilizer([7,8,9,10,11,12]),2,4).stabilizers)


class manual_matrix_stab_generator:
    def __init__(self, num_t6stab, combined_leg_pair):
        self.t6stabilizers = [T6_Stabilizer(range(i * 6, (i + 1) * 6)) for i in range(num_t6stab)]
        self.combined_leg_pair = combined_leg_pair
        self.final_stabilizer = self.combine_stabilizers(0, None)

    def combine_stabilizers(self, pair_idx, current_stabilizer):
        if pair_idx >= len(self.combined_leg_pair):
            return current_stabilizer

        idx1, idx2 = self.combined_leg_pair[pair_idx]
        stab1, leg1 = divmod(idx1 - 1, 6)
        stab2, leg2 = divmod(idx2 - 1, 6)

        if leg1 > 5 or leg2 > 5:
            raise ValueError("Leg index must be 6 or less")

        if current_stabilizer is None:
            current_stabilizer = self.t6stabilizers[stab1]

        # T6_Stabilizer.combineに渡す前に足のインデックスをリストに変換
        current_stabilizer.available_legs = list(current_stabilizer.available_legs)
        self.t6stabilizers[stab2].available_legs = list(self.t6stabilizers[stab2].available_legs)

        combined_stabilizer = current_stabilizer.combine(self.t6stabilizers[stab2], leg1, leg2)
        return self.combine_stabilizers(pair_idx + 1, combined_stabilizer)

# 使用例
num_t6stab = 1
combined_leg_pair = [(1,4)]
#num_t6stab = 3
#combined_leg_pair = [(2, 8), (3, 10), (5, 14)]
generator = manual_matrix_stab_generator(num_t6stab, combined_leg_pair)
final_stabilizer = generator.final_stabilizer
print(final_stabilizer)
# final_stabilizerからcheck_matrixやstabilizersを利用する
check_matrix = final_stabilizer.check_matrix.mat
stabilizers = final_stabilizer.stabilizers
print(check_matrix)
print(stabilizers)