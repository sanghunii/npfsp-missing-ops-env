import random
from typing import List, Tuple

# Customizing Classes
from npfsp_env.modules.job import Job
from npfsp_env.modules.action import ActionSpace
from npfsp_env.modules import setseed


# remove ranodmness 
SEED = setseed.SEED
setseed.set_seed(val=SEED)

## Inspection information (Window based constraints)
INSPECTION_TIME_LIST:List[int] = [25, 50, 75]
INSPECTION_LIMIT = 4    # job 5개당 1개는 inspection을 거쳐야 한다. job 4개까지는 inspection이 강제되지 않으므로 5개중에 1개가 inspection이 강제되는 상황이라면 INSPECTION_LIMIT = 4로 설정된다.


## Cases of batch size in NPFSP. (Not training batch).
EPISODE_LEN_LIST:List[int] = [15, 30, 50, 75, 100, 150]



class Process():    # ENV class
    """
    Args:
        num_machines: 공정 내 기계의 총 갯수
        process_range: 각 machine의 processing time의 범위, Tuple(min processing time, max processing time)으로 주어진다. 
        episode_len_list: cases of episode

    Atrributes:
        buffers: 각 기계별 buffer. (인덱스 = mc_idx) form = List[List[job_numbers]].  e.g. buffers[0][0] : 1번째 machine에 대한 buffer의 첫번째 job의 job_number
        machines: 각 기계의 현재 처리 상태. form = List[List[job_number, remain_process_time]]
        insp_buffers: 각 검사 단계의 buffer. (마지막 기계 제외)
        insps: 검사를 수행하는 단계. remain time 저장.
        finished_jobs: 각 기계별 처리가 끝난 job들의 모음.
    """

    def __init__(self,
                 num_machines: int,
                 process_range: List[Tuple[int, int]],
                 episode_len_list: List[int] = EPISODE_LEN_LIST):
        
        if len(process_range) != num_machines:
            raise ValueError(f"Process range의 길이({len(process_range)})가 machine의 개수{num_machines}와 다릅니다.")
        
        # given values 
        self.num_machines = num_machines
        self.process_range = process_range
        self.episode_len_list: List[int] = episode_len_list
        self.episode_len = 0        # number of job per batch(not training batch).

        # 공정 구성요소
        self.jobsequence: List[Job] = []        # 하나의 jobsequence = 하나의 episode

        # Machine Data (num_machines 개만큼 생성)
        # idx 0: mc1, idx 1: mc2, idx 2: mc3 ...
        self.buffers: List[List[int]] = [[] for _ in range(num_machines)]
        self.input_orders: List[List[int]] = [[] for _ in range(num_machines)]
        self.machines: List[List[int]] = [[] for _ in range(num_machines)]
        self.finished_jobs: List[List[int]] = [[] for _ in range(num_machines)]

        # Inspection Data (마지막 기계 이후엔 검사가 없으므로 num_machines - 1 개 생성)
        self.insp_buffers: List[List[int]] = [[] for _ in range(num_machines - 1)]
        self.insps: List[List[int]] = [[] for _ in range(num_machines - 1)]

        # --- States 동적 할당 ---
        # State 공간 크기 공식: 
        # - 총 크기 = 10 * num_machines - 3 
        self.state_space: int = 10 * num_machines - 3
        
        # State Initialize
        self.state: Tuple[float, ...] = tuple([0.0] * self.state_space)


        # Actions (마지막 기계는 scheduling 대상에서 제외하므로 num_machines - 1 개)
        self.action_space = ActionSpace()
        self.actions: List[int] = [-1] * (num_machines - 1)
        self.insp_actions: List[int] = [-1] * (num_machines - 1)

        # Inspection Limits & Flags (마지막 기계 제외)
        self.inspection_time_list: List[int] = INSPECTION_TIME_LIST
        self.inspection_times: List[int] = [0] * (num_machines - 1)
        self.inspection_limit = INSPECTION_LIMIT
        self.inspection_restricts: List[bool] = [False] * (num_machines - 1)
        self.insp_counters: List[int] = [0] * (self.num_machines - 1)

        # valid machine (마지막 machine 제외)
        if num_machines > 1:
            self.available_machines: List[bool] = [True] + [False] * (num_machines - 2)
        else:
            self.available_machines: List[bool] = []

        # Transition Point
        self.transition_point = False

        # makespan
        self.makespan: int = 0

        # average process time
        self.average_process_time = 0

        # episode endpoint
        self.done = False


    def reset(self, 
              test_inspection_times: List[int] = None, 
              test_process_times: List[List[int]] = None) -> Tuple:
        """
        Args:
            test_inspection_times: 테스트 모드 시 강제로 부여할 inspection time 리스트 (길이: num_machines - 1)
            test_process_times: 테스트 모드 시 강제로 부여할 process time 2차원 리스트 (형태: List[List[int]])
                                e.g., [[mc1_times], [mc2_times], [mc3_times]]
        """
        # 1. 모든 공정 구성요소 초기화
        self.jobsequence: List[Job] = []  

        self.buffers = [[] for _ in range(self.num_machines)]
        self.input_orders = [[] for _ in range(self.num_machines)]
        self.machines = [[] for _ in range(self.num_machines)]
        self.finished_jobs = [[] for _ in range(self.num_machines)]

        self.insp_buffers = [[] for _ in range(self.num_machines - 1)]
        self.insps = [[] for _ in range(self.num_machines - 1)]

        self.insp_counters: List[int] = [0] * (self.num_machines - 1)
        self.inspection_restricts = [False] * (self.num_machines - 1)

        # 2 . 모든 state 초기화
        # __init__에서 정의한 state_space (10 * num_machines - 3) 크기만큼 0.0으로 초기화
        self.state = tuple([0.0] * self.state_space)

        
        # 3, 4. Episode 생성 및 Inspection 정보 초기화
        if (test_inspection_times is not None) and (test_process_times is not None):
            # Test Case (테스트 데이터 주입)

            ## input validation
            if len(test_inspection_times) != self.num_machines - 1:
                raise ValueError(f"test_inspection_times 길이는 {self.num_machines - 1}이어야 합니다.")
            if len(test_process_times) != self.num_machines:
                raise ValueError(f"test_process_times 길이는 {self.num_machines}이어야 합니다.")

            self.inspection_times = test_inspection_times.copy()
            self.episode_len = len(test_process_times[0])

            for i in range(self.episode_len):
                # 우선 빈 Job 생성
                job = Job(num_machines=self.num_machines, job_num=i, process_range=self.process_range)
                
                # 각 기계별로 i번째 작업 시간 추출
                new_times = [test_process_times[m][i] for m in range(self.num_machines)]
                
                # 새로운 update_process_times 메서드 호출 (리스트 전달)
                job.update_process_times(new_times=new_times)
                self.jobsequence.append(job)
        
        else:
            # 3. Generate episode
            self.episode_len = random.choice(self.episode_len_list)
            for i in range(self.episode_len):
                self.jobsequence.append(Job(num_machines=self.num_machines, job_num=i, process_range=self.process_range))

            # 4. inspection restrict관련 정보 초기화 및 inspection times 뽑기
            self.inspection_times = [random.choice(self.inspection_time_list) for _ in range(self.num_machines - 1)]


        # 5-1. 생성된 episode의 average process time 계산 (terminal reward 계산용)
        temp_proc_time = 0
        for job in self.jobsequence:
            temp_proc_time += sum(job.process_times)
        self.average_process_time = temp_proc_time / self.episode_len       

        # 5-2. machine별 모든 job들의 process time 모음
        ## self.all_proc_times[0]: mc1_all_proc
        ## self.all_proc_times[1]: mc2_all_proc ...
        self.all_proc_times = [[] for _ in range(self.num_machines)]
        for job in self.jobsequence:
            for m in range(self.num_machines):
                self.all_proc_times[m].append(job.process_times[m])


        # 6. 모든 job을 첫 번째 기계(mc_idx=0) 버퍼에 투입
        for job in self.jobsequence:
            self.buffers[0].append(job.job_num)
            self.input_orders[0].append(job.job_num)


        # 8. valid machine 관련 정보 초기화 ; 첫번째 machine만 True, 나머지 machine들은 False
        # 마지막 기계는 scheduling 대상에서 제외된다.
        if self.num_machines > 1:
            self.available_machines = [True] + [False] * (self.num_machines - 2)
        else:
            self.available_machines = []

        # 9, 10, 11. 환경 진행 상태 초기화
        self.transition_point = False
        self.makespan = 0
        self.done = False

        # 12. return current process state
        return (self.state, self.done, self.inspection_restricts.copy(), self.available_machines.copy())
    

    def step(self, action_list: List[int]) -> Tuple[Tuple[float, ...], Tuple[int, ...], bool, List[bool], List[bool]]:
        """
        Args:
            action_list: form = List[int]:
                action_list[0] = mc1 action 
                action_list[1] = mc1 insp action . . .

        Return Value:
            ret_type: (next states: Tuple[float], idle_times:Tuple[int], done: bool, inspection_restricts: List[bool], available_machines: List[bool])
        """

        self.done = False
        
        # valid machine 초기화 (스케줄링은 마지막 기계 제외)
        self.available_machines = [False] * (self.num_machines - 1)
        
        # 1 & 2. Action 할당 및 Buffer 정렬 (Apply Dispatching Rule) 
        for m in range(self.num_machines - 1):  # 마지막 machine은 제외한다.
            m_action = action_list[2 * m]
            m_insp_action = action_list[2 * m + 1]

            # 나중에 참고할 수 있도록 저장
            self.actions[m] = m_action
            self.insp_actions[m] = m_insp_action

            if self.buffers[m]:
                if m_action == -1:
                    pass
                elif m_action == 2: # FIFO
                    self.buffers[m] = self.action_space.scheduling(action_num=m_action, jobsequence=self.jobsequence, jobs=self.input_orders[m], mc_num=m+1)
                elif m_action != 6: # Not Inspection & Not FIFO
                    self.buffers[m] = self.action_space.scheduling(action_num=m_action, jobsequence=self.jobsequence, jobs=self.buffers[m], mc_num=m+1)
                elif m_action == 6: # Inspection
                    if m_insp_action == 2:
                        self.buffers[m] = self.action_space.scheduling(action_num=m_insp_action, jobsequence=self.jobsequence, jobs=self.input_orders[m], mc_num=m+1)
                    else:
                        self.buffers[m] = self.action_space.scheduling(action_num=m_insp_action, jobsequence=self.jobsequence, jobs=self.buffers[m], mc_num=m+1)
                else:
                    raise Exception(f"mc{m+1}_action에 올바르지 않은 action number가 할당됨: {m_action}")


        # 3. 정렬된 것을 토대로 각 기계에 Job 할당
        for m in range(self.num_machines):
            if not self.machines[m] and self.buffers[m]:    # 현재 machine이 empty상태이고, 해당 machine의 buffer에 투입가능한 job이 있다면 
                if m < self.num_machines - 1 and self.actions[m] == -1:
                    # Last machine은 scheduling 대상에서 제외되므로 last machine의 action은 체크할 필요 없음.
                    raise Exception(f"mc{m+1}이 할당 가능한 상태인데 action이 -1입니다.")

                
                # 기계에 Job 할당
                job_num = self.buffers[m][0]        # 이번에 할당할 job의 job_number가져오기
                self.machines[m] = [job_num, self.jobsequence[job_num].process_times[m]]
                self.buffers[m].pop(0)
                self.input_orders[m].remove(job_num)

                # Inspection constraints
                if m < self.num_machines - 1:   # 마지막 machine 제외.
                    # Insp validation ; Inspection이 강제되는 상황에서 insp action이 선택되지 않았다면 error
                    if self.inspection_restricts[m] and self.actions[m] != 6:
                        raise ValueError(f"mc{m+1}은 검사 강제 상태(restrict=True)인데, 검사 액션(6)이 아닌 {self.actions[m]}이(가) 입력되었습니다.")
                    
                    # m번째 machine에서 inspection action이 선택 됐을 때.
                    if self.actions[m] == 6:
                        self.jobsequence[job_num].set_inspection(stage_idx=m, inspection_time=self.inspection_times[m])
                        self.insp_counters[m] = 0
                        self.inspection_restricts[m] = False
                    # m번째 machine에서 inspection actino이 선택 되지 않았을 때.
                    else:
                        self.insp_counters[m] += 1
                        if self.insp_counters[m] == self.inspection_limit:
                            self.inspection_restricts[m] = True


        # 4. Processing (다음 transition point가 될 때까지 공정 과정 진행)
        idle_times_list = [0] * (self.num_machines - 1)     # Idle Time 초기화 (1번 기계 제외, mc2 ~ last_machine 의 개수만큼)
        self.transition_point = False

        while not self.transition_point:
            # 4-a. 최단 Process Time 계산
            process_time_list = []
            for m in range(self.num_machines):
                if self.machines[m] and self.machines[m][1] != 0:
                    process_time_list.append(self.machines[m][1])
                if m < self.num_machines - 1 and self.insps[m] and self.insps[m][1] != 0:
                    process_time_list.append(self.insps[m][1])

            process_time = min(process_time_list)
            self.makespan += process_time
            
            # 4-b. Remain Time 갱신
            for m in range(self.num_machines):
                if self.machines[m] and self.machines[m][1] != 0:
                    self.machines[m][1] -= process_time
                if m < self.num_machines - 1 and self.insps[m] and self.insps[m][1] != 0:
                    self.insps[m][1] -= process_time
            
            # 4-c. Idle Time 계산 (mc2 부터 마지막 기계까지)  
            # Total Idle Time (Front + Middle + Residual)
            for m in range(1, self.num_machines):
                if not self.machines[m] and not self.buffers[m]:
                    idle_times_list[m-1] += process_time
            
            # 4-d. Transition Point 체크 및 이동 로직
            # --- Inspection Check: 각 Inspection machine먼저 체크하고 끝난 애들 다음 machine의 buffer로 넣어준다. ---
            for m in range(self.num_machines - 1):
                if self.insps[m]:
                    if self.insps[m][1] == 0:
                        job_num = self.insps[m][0]
                        self.buffers[m+1].append(job_num)
                        self.input_orders[m+1].append(job_num)
                        self.insps[m].clear()
                        
                        if self.insp_buffers[m]:
                            next_job = self.insp_buffers[m][0]
                            self.insps[m] = [next_job, self.jobsequence[next_job].inspection_times[m]]
                            self.insp_buffers[m].pop(0)
                
                elif not self.insps[m] and self.insp_buffers[m]:
                    """
                    현재 Process class 구현에서는 해당 block이 호출될 일은 없다.
                    왜냐면 현재 구현상 insp machine이 투입가능한 상태면 바로바로 job을 투입해주기 때문에
                    insp machine이 empty상태이고 동시에 해당 insp machine의 buffer에 투입가능한 job이 있을 수 없다.
                    """
                    next_job = self.insp_buffers[m][0]
                    self.insps[m] = [next_job, self.jobsequence[next_job].inspection_times[m]]
                    self.insp_buffers[m].pop(0)

            # --- Machine Check: 개별 machine체크하고 투입 가능한 상태인 machine을 찾는다. ---
            for m in range(self.num_machines):
                if self.machines[m]:
                    if self.machines[m][1] == 0:
                        # 해당 machine의 작업이 끝난 상황일 때
                        job_num = self.machines[m][0]       # 끝난 job의 번호
                        self.finished_jobs[m].append(job_num)   # 해당 machine의 종료 작업 목록에 추가.

                        if m < self.num_machines - 1: # 중간 기계들 (mc1, mc2...)       # last machine이 아니라면 => 다음 단계의 machine으로 옮겨야함 (mc or insp_mc)
                            if self.jobsequence[job_num].inspection_times[m] != 0:
                                 # inspection을 거치는 job이라면.
                                if not self.insps[m] and not self.insp_buffers[m]:
                                    self.insps[m] = [job_num, self.jobsequence[job_num].inspection_times[m]]
                                    self.machines[m].clear()
                                else:
                                    self.insp_buffers[m].append(job_num)
                                    self.machines[m].clear()
                            else:
                                # inspection 거치지 않는 job이라면.
                                self.buffers[m+1].append(job_num)
                                self.input_orders[m+1].append(job_num)
                                self.machines[m].clear()

                            if len(self.finished_jobs[m]) != self.episode_len and self.buffers[m]:
                                # 해당 machine에서 처리해야 하는 job이 남았고, 현재 buffer에 투입 대기중인 job이 있다면
                                self.transition_point = True
                                self.available_machines[m] = True
                                
                        else:
                            # Last machines
                            self.machines[m].clear()
                            if len(self.finished_jobs[m]) == self.episode_len:
                                # Episode end point
                                self.done = True
                                self.transition_point = True
                            elif self.buffers[m]:
                                # End point가 아니고 buffer에 투입 가능한 job이 있다면
                                # Last machine은 scheduling대상에서 제외되므로 그때그때 바로 투입.
                                next_job = self.buffers[m][0]
                                self.machines[m] = [next_job, self.jobsequence[next_job].process_times[m]]
                                self.buffers[m].pop(0)
                                self.input_orders[m].remove(next_job)

                # 할당 지연 (Delayed Allocation) 상황 처리
                elif not self.machines[m] and self.buffers[m]:
                    if m < self.num_machines - 1: 
                        self.transition_point = True
                        self.available_machines[m] = True
                    else: 
                        # 마지막 기계는 스케줄링 대상이 아니므로 고민 없이 즉시 할당
                        job_num = self.buffers[m][0]
                        self.machines[m] = [job_num, self.jobsequence[job_num].process_times[m]]
                        self.buffers[m].pop(0)
                        self.input_orders[m].remove(job_num)


        # 5. 동적 State 계산 로직
        """
        REMOVED
        """

        # 최종 Tuple 변환
        idle_times = tuple(idle_times_list)

        return (self.state, idle_times, self.done, self.inspection_restricts.copy(), self.available_machines.copy())
    
    def close(self):
        self.reset()
        print(f"\n\n{'*'*10} 환경 사용이 종료되었습니다. {'*'*10}")


    def __str__(self):
        """
        Process의 전체적인 상황을 간단하게 설명:
            각 기계(및 검사기)의 buffer와 현재 할당 상태, 그리고 최종 완료된 job들을 출력하낟.
        """
        result = []
        for m in range(self.num_machines):
            result.append(f"mc{m+1} buffer: {self.buffers[m]}")
            result.append(f"mc{m+1}: {self.machines[m]}\n")
            
            # 마지막 기계가 아니면 Inspection 정보도 추가
            if m < self.num_machines - 1:
                result.append(f"mc{m+1}_insp buffer: {self.insp_buffers[m]}")
                result.append(f"mc{m+1}_insp: {self.insps[m]}\n")
        
        # 전체 공정이 끝난 작업은 마지막 기계의 finished_jobs를 확인
        result.append(f"finished job: {self.finished_jobs[-1]}")
        
        return "\n".join(result)


    def __repr__(self):
        """
        Debugging용 상세 설명:
            기계 개수에 맞춰 동적으로 모든 job의 상세 정보, action, makespan, done 상태를 출력한다.
        """
        result = []
        
        # 1. Action Information (마지막 기계 제외)
        result.append("\n\n<<선택된 action>>")
        for m in range(self.num_machines - 1):
            result.append(f"machine{m+1}: {self.actions[m]}")
            
        result.append(f"\nNow Available Machine: {self.available_machines}\n")
        result.append(f"{'-'*40} action에 따른 결과 {'-'*40}")

        # 2. Machine & Inspection Information
        for m in range(self.num_machines):
            # Main Machine Info
            result.append(f"<<mc{m+1} origin order >>\n{self.input_orders[m]}")
            
            buf_info = "\n".join([self.jobsequence[j].__repr__() for j in self.buffers[m]]) if self.buffers[m] else "empty"
            result.append(f"<<mc{m+1} buffer>>\n{buf_info}")
            
            if self.machines[m]:
                mc_info = f"{self.jobsequence[self.machines[m][0]].__repr__()}\nmachine{m+1} remain process time: {self.machines[m][1]}"
            else:
                mc_info = "empty"
            result.append(f"<<<mc{m+1}>>>\n{mc_info}\n")
            
            # Inspection Info (마지막 기계 제외)
            if m < self.num_machines - 1:
                insp_buf_info = "\n".join([self.jobsequence[j].__repr__() for j in self.insp_buffers[m]]) if self.insp_buffers[m] else "empty"
                result.append(f"<<<mc{m+1} inspection buffer>>>\n{insp_buf_info}")
                
                if self.insps[m]:
                    insp_info = f"{self.jobsequence[self.insps[m][0]].__repr__()}\ninspection remain process time: {self.insps[m][1]}"
                else:
                    insp_info = "empty"
                result.append(f"<<<mc{m+1} inspection>>>\n{insp_info}\n")

        # 3. Finished Job Information
        for m in range(self.num_machines):
            fin_info = "\n".join([self.jobsequence[j].__repr__() for j in self.finished_jobs[m]]) if self.finished_jobs[m] else ""
            if fin_info: # 끝난 작업이 있을 때만 출력
                result.append(f"<<mc{m+1}_finished job>>\n{fin_info}\n")

        # 4. Global Info (Makespan, Counters, Done)
        result.append(f"<<makespan>>\n{self.makespan}\n")
        
        for m in range(self.num_machines - 1):
            result.append(f"<<mc{m+1} inspection counter>>\n{self.insp_counters[m]}")
            
        result.append(f"\n<<done>>\n{self.done}\n")

        return "\n".join(result)