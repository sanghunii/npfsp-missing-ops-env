from typing import List
from npfsp_env.modules.job import Job

class ActionSpace:
    """
    Actions
        2: FIFO (First In First Out)
        
    Methods
        - scheduling(): 선택된 action(rule)에 따라 특정 머신 버퍼의 job들을 우선순위대로 정렬.
        - sample(): Exploration을 위한 랜덤 action 반환. 
            * caution: inspection action은 -1로 고정되며, 의사결정 시점이 아닌 머신의 마스킹 처리는 외부(호출부)에서 담당해야 함.
    """

    def __init__(self):
        self.num_pdrs = 1

    ## Public
    def scheduling(self, action_num: int, jobsequence: 'List[Job]', jobs: List[int], mc_num: int) -> List[int]:
        # 사용자의 mc_num은 1부터 시작한다고 가정 (mc1 -> 1, mc2 -> 2)
        # 리스트 인덱싱을 위해 0부터 시작하는 index로 변환
        mc_idx = mc_num - 1

        if action_num == 2:
            return self._fifo(mc_idx=mc_idx, jobsequence=jobsequence, jobs=jobs)
        else:
            raise ValueError(f"유효하지 않은 action_num 입니다: {action_num}")
        

    
    def sample(self, num_machines: int) -> List[int]:
        """
        Exploration시 각 machine에 대해 랜덤한 action을 선택하여 반환.
        """
        actions = []
        valid_machines = num_machines - 1           # 마지막 machine은 scheduling대상에서 제외 (현재 우리 환경에서는 cmax에 영향 x)
        
        for _ in range(valid_machines):
            """"""
            mc_action = 2
            mc_insp_action = -1
            
            actions.extend([mc_action, mc_insp_action])
            
        return actions
    

    ## Private Attrs
    def _fifo(self, mc_idx: int, jobsequence: 'List[Job]', jobs: List[int]) -> List[int]:
        """
        fifo를 사용할때는 jobs에 해당 machine의 input order가 들어와야 한다.
        """
        return [job_num for job_num in jobs]
