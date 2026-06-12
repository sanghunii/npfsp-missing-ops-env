import random
from typing import List, Tuple

class Job:
    """
    Episoe(JobSequence)를 구성하는 개별 Job의 형태를 정의한다.
    num_machines개의 기계와 (num_machines-1)개의 inspection으로 구성된 Process를 가진다.

    Args:
        num_machines: 공정을 구성하는 기계의 총 개수
        job_num : 해당 job의 번호를 매긴다. 번호를 주지 않으면 0번 job으로 만들어진다.
        process_range : 각 기계별 (min, max) process time 범위를 담은 리스트
    """

    def __init__(self, 
                 num_machines: int, 
                 job_num: int = 0, 
                 process_range: List[Tuple[int, int]] = None):
        
        self.num_machines = num_machines
        self.job_num = job_num

        if process_range is None or len(process_range) != num_machines:
            raise ValueError("process_range의 길이는 num_machines와 같아야 합니다.")

        # 1. Machine Process Times (리스트로 동적 생성)
        self.process_times: List[int] = [
            random.randint(pr[0], pr[1]) for pr in process_range
        ]

        # 2. Inspection Process Times 
        # 마지막 기계 이후에는 inspection이 없으므로 num_machines - 1 개 생성
        self.inspection_times: List[int] = [0] * (num_machines - 1)

        # 초기값 계산
        self._calculate_derived_values()


    def set_inspection(self, stage_idx: int, inspection_time: int):
        """
        특정 단계의 inspection time 설정
        stage_idx: 0이면 첫 번째(MC1 직후) inspection, 1이면 두 번째(MC2 직후) inspection
        """
        if 0 <= stage_idx < self.num_machines - 1:
            self.inspection_times[stage_idx] = inspection_time
        else:
            raise IndexError("유효하지 않은 Inspection 단계입니다.")
        

    def update_process_times(self, new_times: List[int]):
        """테스트 데이터로 처리 시간을 변경할 때, u_group과 rmr을 함께 재계산"""
        if len(new_times) != self.num_machines:
            raise ValueError(f"새로운 process time 리스트의 길이는 {self.num_machines}이어야 합니다.")
        
        self.process_times = new_times
        # 값을 바꾼 후 파생 변수들(u_group, rmr) 일괄 재계산
        self._calculate_derived_values()

    
    def _calculate_derived_values(self):
        """Process time이 결정/변경될 때마다 u_group과 rmr을 동적으로 재계산"""
        for i in range(self.num_machines - 1):
            pass


    def __str__(self) -> str:
        # 출력 포맷을 동적으로 생성
        proc_str = "   ".join([f"mc{i+1} process time : {self.process_times[i]}" for i in range(self.num_machines)])
        
        insp_str = "   ".join([f"Inspection{i+1} : {self.inspection_times[i] if self.inspection_times[i] != 0 else False}" 
                               for i in range(self.num_machines - 1)])
        

        return (f"job number : {self.job_num}   "
                f"{proc_str}   "
                f"{insp_str}   ")

    def __repr__(self) -> str:
        return self.__str__()