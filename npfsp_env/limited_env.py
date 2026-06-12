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
INSPECTION_LIMIT = 4

## Cases of batch size in NPFSP. (Not training batch).
EPISODE_LEN_LIST:List[int] = [15, 30, 50, 75, 100]  # WO size150

class Process():    # ENV class
    """
    Args:
        num_machines: 공정 내 기계의 총 갯수
        process_range: 각 machine의 processing time의 범위, Tuple(min processing time, max processing time)으로 주어진다. 
        episode_len_list: cases of episode
        buffer_limit: [NEW] 기계(mc2 부터) 및 검사대의 최대 버퍼 용량 제한. mc1은 무한 버퍼로 동작.

    Atrributes (Refactoring ver):
        buffers: 각 기계별 buffer. (인덱스 = mc_idx) form = List[List[job_numbers]].  e.g. buffers[0][0] : 1번째 machine에 대한 buffer의 첫번째 job의 job_number
        machines: 각 기계의 현재 처리 상태. form = List[List[job_number, remain_process_time]]
        insp_buffers: 각 검사 단계의 buffer. (마지막 기계 제외)
        insps: 검사를 수행하는 단계. remain time 저장.
        finished_jobs: 각 기계별 처리가 끝난 job들의 모음.
    """

    def __init__(self,
                 num_machines: int,
                 process_range: List[Tuple[int, int]],
                 episode_len_list: List[int],
                 buffer_limit: int):
        
        if len(process_range) != num_machines:
            raise ValueError(f"Process range의 길이({len(process_range)})가 machine의 개수{num_machines}와 다릅니다.")
        
        # given values 
        self.num_machines = num_machines
        self.process_range = process_range
        self.episode_len_list: List[int] = episode_len_list
        self.episode_len = 0        # number of job per batch(not training batch).
        self.buffer_limit = buffer_limit

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
        # State 공간 크기 공식 (Limited Buffer 업데이트 반영): 
        self.state_space: int = 13 * self.num_machines - 6
        
        # State 초기화: 모든 값을 0.0으로 가지는 튜플 생성
        self.state: Tuple[float, ...] = tuple([0.0] * self.state_space)


        # Actions (마지막 기계는 scheduling 대상에서 제외하므로 num_machines - 1 개)
        self.action_space = ActionSpace()
        self.actions: List[int] = [-1] * (num_machines - 1)
        self.insp_actions: List[int] = [-1] * (num_machines - 1)

        # Inspection Limits & Flags (마지막 기계 제외)
        self.inspection_time_list: List[int] = INSPECTION_TIME_LIST
        self.inspection_times: List[int] = [0] * (num_machines - 1)
        self.inspection_limit = INSPECTION_LIMIT    # inspection constratins window size에 따른 최대 연속으로 inspection을 건너뛸 수 있는 횟수
        self.inspection_restricts: List[bool] = [False] * (num_machines - 1)
        self.insp_counters: List[int] = [0] * (self.num_machines - 1)

        # valid machine (마지막 machine 제외)
        # 초기 상태에서는 1번 기계(인덱스 0)만 True, 나머지는 False라고 가정 (기존 [True, False] 로직 반영)
        if num_machines > 1:
            self.available_machines: List[bool] = [True] + [False] * (num_machines - 2)
        else:
            self.available_machines: List[bool] = []

        # Transition Point
        self.transition_point = False

        # makespan
        self.makespan: int = 0

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
        # 1. 모든 공정 구성요소 초기화 (동적 리스트 클리어)
        self.jobsequence: List[Job] = []  

        self.buffers = [[] for _ in range(self.num_machines)]
        self.input_orders = [[] for _ in range(self.num_machines)]
        self.machines = [[] for _ in range(self.num_machines)]
        self.finished_jobs = [[] for _ in range(self.num_machines)]

        self.insp_buffers = [[] for _ in range(self.num_machines - 1)]
        self.insps = [[] for _ in range(self.num_machines - 1)]

        self.insp_counters: List[int] = [0] * (self.num_machines - 1)
        self.inspection_restricts = [False] * (self.num_machines - 1)

        # 2 . 모든 state 초기화 (0.0으로 이루어진 동적 튜플 생성)
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

        # 5. machine별 모든 job들의 process time 모음
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
        
        self.state = self._get_state()

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


        # 3. 정렬된 것을 토대로 각 기계에 Job 할당 및 0초 연쇄 막힘 해소 (역순 진행)
        for m in range(self.num_machines - 1, -1, -1):
            
            # 3-A. m번째 기계가 비어있고, buffer에는 대기중인 job이 있을 때.
            if not self.machines[m] and self.buffers[m]:    

                # 마지막 machine이 아닌 경우.
                if m < self.num_machines - 1:
                    if self.actions[m] == -1:
                        raise Exception(f"mc{m+1}이 할당 가능한 상태인데 action이 -1입니다.")
                    
                    job_num = self.buffers[m][0]
                    self.machines[m] = [job_num, self.jobsequence[job_num].process_times[m]]
                    self.buffers[m].pop(0)
                    self.input_orders[m].remove(job_num)

                    # Inspection constraints 검사 m번째 machine에서 inspection이 강제되는 상황인데 inspection action이 선택되지 않았다는 것은 에러임.
                    if self.inspection_restricts[m] and self.actions[m] != 6:
                        raise ValueError(f"mc{m+1}은 검사 강제 상태...")
                    
                    # Inspection action 처리. 이때의 agent의 inspection action선택은 제약조건에 의한 선택일 수도 있고 bypass로서의 활용을 위한 자발적인 선택일 수도 있다.
                    if self.actions[m] == 6:
                        self.jobsequence[job_num].set_inspection(stage_idx=m, inspection_time=self.inspection_times[m])
                        self.insp_counters[m] = 0
                        self.inspection_restricts[m] = False

                    # 그 외의 action들 처리.
                    else:
                        self.insp_counters[m] += 1
                        if self.insp_counters[m] == self.inspection_limit:
                            self.inspection_restricts[m] = True

            
            # 3-B. 직전 machine이나 직전 inspection machine에 blocking현상으로 인해 못 넘어온 job이 있으면 갖고오기
            if m > 0:
                prev_m = m - 1  
                
                # 직전 inspection machine부터 먼저 blocking된 job이 있는지 확인 (우선순위)
                if self.insps[prev_m] and self.insps[prev_m][1] == 0:
                    if len(self.buffers[m]) < self.buffer_limit: 
                        job_num = self.insps[prev_m][0]
                        self.buffers[m].append(job_num)
                        self.input_orders[m].append(job_num)
                        self.insps[prev_m].clear()

                        if self.insp_buffers[prev_m]:
                            next_job = self.insp_buffers[prev_m][0]
                            self.insps[prev_m] = [next_job, self.jobsequence[next_job].inspection_times[prev_m]]
                            self.insp_buffers[prev_m].pop(0)

                # 직전 machine에 blocking된 job이 있는지 확인
                if self.machines[prev_m] and self.machines[prev_m][1] == 0:
                    job_num = self.machines[prev_m][0]

                    # 1. m-1번째 machine -> m-1 inspection machine 해당 과정에서 blocking된 job.
                    if self.jobsequence[job_num].inspection_times[prev_m] != 0:
                        if len(self.insp_buffers[prev_m]) < self.buffer_limit: 
                            self.insp_buffers[prev_m].append(job_num)
                            self.machines[prev_m].clear()
                            self.finished_jobs[prev_m].append(job_num)

                    # 2. m-1번째 machine -> m번째 machine (현재 job투입이 일어난 machine) 해당 과정에서 blocking된 job.
                    else:
                        if len(self.buffers[m]) < self.buffer_limit: 
                            self.buffers[m].append(job_num)
                            self.input_orders[m].append(job_num)
                            self.machines[prev_m].clear()
                            self.finished_jobs[prev_m].append(job_num)



        # 4. Processing (다음 transition point가 될 때까지 공정 과정 진행)
        idle_times_list = [0] * (self.num_machines - 1)         # 1번째 machine 제외.
        blocked_times_list = [0] * (self.num_machines - 1)      # last machine 제외.
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

            # 4-b. Idle Time & Blocked Time 계산 (for reward)
            for m in range(self.num_machines):
                if m > 0 and not self.machines[m] and not self.buffers[m]:
                    idle_times_list[m-1] += process_time
                
                if m < self.num_machines - 1:
                    if self.machines[m] and self.machines[m][1] == 0:
                        blocked_times_list[m] += process_time
            
            # 4-c. Remain Time 갱신
            for m in range(self.num_machines):
                if self.machines[m] and self.machines[m][1] != 0:
                    self.machines[m][1] -= process_time
                if m < self.num_machines - 1 and self.insps[m] and self.insps[m][1] != 0:
                    self.insps[m][1] -= process_time
            

            # 4-d. 작업이 종료된 job들 이동 및 transition point & available machine check 로직
            for m in range(self.num_machines - 1, -1, -1):

                # 1. Last Machine Logic
                if m == self.num_machines - 1:

                    # last machine에서 작업이 종료되었을때
                    if self.machines[m] and self.machines[m][1] == 0:
                        job_num = self.machines[m][0]
                        self.finished_jobs[m].append(job_num)
                        self.machines[m].clear()
                    
                    # last machine에서 마지막 job이 끝났을 때 => episode end point
                    if len(self.finished_jobs[m]) == self.episode_len:
                        self.done = True
                        self.transition_point = True
                    
                    # 마지막 작업이 끝난게 아니고 당장 buffer에 투입가능한 job이 있다면 바로 투입해주기.
                    elif not self.machines[m] and self.buffers[m]:
                        self.buffers[m] = self.action_space.scheduling(
                            action_num=2, jobsequence=self.jobsequence, jobs=self.buffers[m], mc_num=self.num_machines
                        )
                        next_job = self.buffers[m][0]
                        self.machines[m] = [next_job, self.jobsequence[next_job].process_times[m]]
                        self.buffers[m].pop(0)
                        self.input_orders[m].remove(next_job)
                    continue 


                # 이번 턴에 m+1의 available_machines[m+1]을 true로 설정할지 스위치
                trigger_wakeup = False

                # 2. Inspection Check (물리 이동)
                # inspection이 딱 끝났을 때,
                if self.insps[m] and self.insps[m][1] == 0:

                    # m+1번째 machine buffer에 여유가 있다면.
                    if len(self.buffers[m+1]) < self.buffer_limit:      
                        job_num = self.insps[m][0]
                        
                        # m+1번째 machine buffer가 텅 빈 상태라면. ; m+1번째 machine의 상태 flag를 갱신할지 말지 결정해야한다.
                        if len(self.buffers[m+1]) == 0: 
                            trigger_wakeup = True
                        
                        # 일단 m+1번째 machine buffer에다가 job할당.
                        self.buffers[m+1].append(job_num)
                        self.input_orders[m+1].append(job_num)
                        self.insps[m].clear()

                        # inspection machine buffer에 대기중인 job이 있다면 바로 투입.
                        if self.insp_buffers[m]:    
                            next_job = self.insp_buffers[m][0]
                            self.insps[m] = [next_job, self.jobsequence[next_job].inspection_times[m]]
                            self.insp_buffers[m].pop(0)


                # 3. Machine Check 
                # m번째 machine에서의 작업이 끝났다면 
                if self.machines[m] and self.machines[m][1] == 0:
                    job_num = self.machines[m][0]       

                    # 이번에 끝난 job이 inspection을 거쳐야하는 job이라면 (by forced or for bypass)
                    if self.jobsequence[job_num].inspection_times[m] != 0:
                        # m번째 inspection machine이 유휴상태일 때면.
                        if not self.insps[m] and not self.insp_buffers[m]:
                            self.insps[m] = [job_num, self.jobsequence[job_num].inspection_times[m]]
                            self.machines[m].clear()
                            self.finished_jobs[m].append(job_num)
                        
                        # m번째 inspection machine이 유휴상태는 아니지만 buffer는 비어있을 때
                        elif len(self.insp_buffers[m]) < self.buffer_limit: 
                            self.insp_buffers[m].append(job_num)
                            self.machines[m].clear()
                            self.finished_jobs[m].append(job_num)

                        # inspection machine에 자리가 없을때: blocking
                        else:
                            pass
                    
                    # 이번에 끝난 job이 inspection을 거치지 않는 job이라면 => m+1번째 machine에다가 할당해야한다.
                    else:
                        # m+1번째 기계에서 job투입이 일어나서 buffer에서 한자리가 빌 예정인지 확인.
                        b_m1_will_open = self.available_machines[m+1] if m + 1 < self.num_machines - 1 else (len(self.buffers[m+1]) < self.buffer_limit or not self.machines[m+1])

                        # m+1번째 machine buffer에 자리가 있을 때
                        if len(self.buffers[m+1]) < self.buffer_limit: 
                            # 이때 m+1 buffer가 완전히 빈 상태라면 flag를 true로 갱신해줘야할지 말지 결정해 줘야한다.
                            if len(self.buffers[m+1]) == 0: 
                                trigger_wakeup = True 
                            
                            # m+1번째 machine buffer에다가 job할당.
                            self.buffers[m+1].append(job_num)
                            self.input_orders[m+1].append(job_num)
                            self.machines[m].clear()
                            self.finished_jobs[m].append(job_num)   
                        
                        # m+1번째 machine에서 다음번에 바로 job하나가 빠져서 자리가 날 예정이고, m번째 inspection에서 m번째 machine으로 job을 할당할 예정이 없으면 일시적인 blocking. 
                        # 이때 m번째 machine의 buffer에 대기 중인 job이 있거나 들어올 예정이라면 available_machines[m]
                        elif b_m1_will_open and not (self.insps[m] and self.insps[m][1] == 0):
                            pass 

                        # 둘 다 해당이 안된다면 그냥 blocking상황
                        else:
                            pass # 완전히 Blocked 


                # 4. wake-up 플래그 갱신
                # m번째 machine(혹은 inspection machine)을 체크하던 중 trigger_wakeup이 true => m+1번째 machine으로 job이 할당되었는데 해당 machine의 buffer가 비어있었다. 
                # 이때 m+1이 blocking상황이 아니라면 m+1번째 machine에서도 의사결정이 일어나야 하므로 available_machiens[m+1] = True로 만들어줘야 하는지 아닌지 체크해줘야 한다.
                # 즉, 해당 로직은 m번째 machine을 검사하고 있을 때 m+1번째 machine의 flag를 적절하게 update하기 위한 로직이다.
                if trigger_wakeup:
                    # m+1번째 기계가 마지막 기계일 때
                    if m + 1 == self.num_machines - 1:
                        if not self.machines[m+1]:
                            
                            self.buffers[m+1] = self.action_space.scheduling(
                                action_num=2, jobsequence=self.jobsequence, jobs=self.buffers[m+1], mc_num=self.num_machines
                            )
                            next_job = self.buffers[m+1][0]
                            self.machines[m+1] = [next_job, self.jobsequence[next_job].process_times[m+1]]
                            self.buffers[m+1].pop(0)
                            self.input_orders[m+1].remove(next_job)

                    # m+1번째 기계가 마지막 기계가 아닐 때.
                    else:
                        # m+1번째 machine이 비어있을 때
                        if not self.machines[m+1]:
                            # m+1번째 machine에 대기중인 job이 있을 때
                            if self.buffers[m+1]:
                                self.transition_point = True
                                self.available_machines[m+1] = True
                        
                        # m+1번째 machine에 현재 blocking당한 job이 있을 때
                        # 해당 blocking당한 job이 m+2번째 machine 혹은 m+1번째 inspection machine으로 빠질 예정인지 확인해야한다.
                        # 빠질 예정이라면 available_machines[m+1]을 true로 설정해줘야한다.
                        elif self.machines[m+1][1] == 0:
                            b_m2_will_open = self.available_machines[m+2] if m + 2 < self.num_machines - 1 else (len(self.buffers[m+2]) < self.buffer_limit or not self.machines[m+2])
                            
                            # m+2번째 machine에 빈자리가 생긴다고 판단이 된다면
                            if b_m2_will_open:
                                # m+1번째 machine의 blocked job이 inspection을 거쳐야 하는 job인지 아닌지 판단. 
                                blocked_job = self.machines[m+1][0]
                                needs_insp = (self.jobsequence[blocked_job].inspection_times[m+1] != 0)
                                
                                # m+1번째 machine의 blocked job이 다음 단계로 넘어갈지 아닐지 판단.
                                next_will_pull = False

                                # blocked job이 inspection을 거쳐야 하는 job이라면 
                                if needs_insp:
                                    # m+1번째 inspectoin machine이 현재 비어있거나 0s여야 m+1의 blocked job도 연쇄적으로 바로 빠질 예정이다.
                                    if not self.insps[m+1] or (self.insps[m+1] and self.insps[m+1][1] == 0):
                                        next_will_pull = True
                                
                                # blocked job이 inspection을 거치지 않는 job이라면
                                else:
                                    # 이미 앞서서 m+2번째 machine buffer에는 빈자리가 생김을 확인했다. 이런 경우엔 m+1 inspection machine에서 m+2 machine으로 job이 넘어가는 것이 예정되어 있지 않다면 m+1번째 machine의 blocking이 해제된다.
                                    if not (self.insps[m+1] and self.insps[m+1][1] == 0):
                                        next_will_pull = True

                                # m번째 machine 및 m번째 inspection machine검사 과정에서 empty상태의 m+1번째 machine buffer에 job이 할당되었고, m+1번째 machine의 blocked job이 빠져나갈 예정이라면. transition point 및 available_machiens[m+1] flag= true로 갱신
                                if next_will_pull and self.buffers[m+1]:
                                    self.transition_point = True
                                    self.available_machines[m+1] = True


                # 5. 현재 검사중인 m번째 machien의 flag를 적절하게 설정하는 로직이다.
                # m번째 machine이 비어있고 buffer에 대기중인 job이 있을 때
                if not self.machines[m] and self.buffers[m]:
                    self.transition_point = True
                    self.available_machines[m] = True
                        
                elif self.machines[m] and self.machines[m][1] == 0 and self.buffers[m]:
                    
                    # buffer of m+1 machine가 열릴 예정인가?
                    b_m1_will_open = self.available_machines[m+1] if m + 1 < self.num_machines - 1 else (len(self.buffers[m+1]) < self.buffer_limit or not self.machines[m+1])
                    
                    blocked_job = self.machines[m][0]
                    needs_insp = (self.jobsequence[blocked_job].inspection_times[m] != 0)
                    
                    next_will_pull = False

                    # m번째 machine에서 끝난 job이 inspection을 거치는 job이라면
                    if needs_insp:
                        # 검사 버퍼에 1자리라도 비어있으면 뒤(m+1)가 막히든 말든 무조건 탈출
                        if len(self.insp_buffers[m]) < self.buffer_limit:
                            next_will_pull = True
                            
                        # 검사 버퍼 꽉 참 + 검사기 0초 대기 중
                        # 이 경우 반드시 "다음 메인 버퍼(b_m1_will_open)가 열려야 m번째 machine의 blocked job이 다음으로 넘어갈 수 있다.
                        elif self.insps[m] and self.insps[m][1] == 0 and b_m1_will_open:
                            next_will_pull = True
                            
                    # m번째 machine에서 끝난 job이 inspection을 거치지 않는 job이라면.
                    else: 
                        # 다음 메인 버퍼가 열려야 하고, 검사기가 새치기하지 않아야 탈출
                        # 다음 메인 버퍼가 열려야 하고, m번째 inspection machine에서 m+1번째 machine으로 할당 예정인 job이 없으면 투입 가능.
                        if b_m1_will_open:
                            if not (self.insps[m] and self.insps[m][1] == 0):
                                next_will_pull = True
                            
                    if next_will_pull:
                        self.transition_point = True
                        self.available_machines[m] = True



        # 5. 동적 State 계산 로직
        self.state = self._get_state()


        # idle times와 blocked times 최종 Tuple 변환
        idle_times = tuple(idle_times_list)
        blocked_times = tuple(blocked_times_list)

        return (self.state, idle_times, blocked_times, self.done, self.inspection_restricts.copy(), self.available_machines.copy())
    



    def close(self):
        self.reset()
        print(f"\n\n{'*'*10} 환경 사용이 종료되었습니다. {'*'*10}")

    

    def _get_state(self) -> Tuple[float, ...]:
        """
        연구실 내부 사정상 state로직 가림. 
        """
        states = self.state

        return tuple(states)



    def __str__(self):
        """
        Process의 전체적인 상황을 간단하게 설명:
            각 기계(및 검사기)의 buffer와 현재 할당 상태,
            그리고 최종 완료된 job들을 출력합니다.
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
            기계 개수에 맞춰 동적으로 모든 job의 상세 정보, action, makespan, done 상태를 출력합니다.
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
