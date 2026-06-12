import random
from typing import List
from npfsp_env import limited_env

def main():
    episode_num = 100
    temp_episode_len_list = [100]

    # ENV PARAMETERS
    MC_NUM = 3
    PROCESS_RANGE = [(20, 90)] * MC_NUM
    BUFFER_LIMIT = 3
    
    print(f"{'=' * 50}")
    print("🚀 NPFSP Environment Test Started...")
    print(f"{'=' * 50}\n")
    
    env = limited_env.Process(
        num_machines=MC_NUM, 
        process_range=PROCESS_RANGE, 
        episode_len_list=temp_episode_len_list, 
        buffer_limit=BUFFER_LIMIT
    )
    
    step_counter = 0
    makespan_list = []
    
    total_idle_times = [0] * (MC_NUM - 1)
    total_blocked_times = [0] * (MC_NUM - 1)
    
    for i in range(episode_num):
        # 1. Reset Environment
        current_state, done, inspection_restricts, available_machines = env.reset()
        branch_num = MC_NUM - 1 

        while not done:
            step_counter += 1
            actions: List[int] = []
            
            for m_idx in range(branch_num):
                mc_action = -1
                mc_insp_action = -1
                
                if available_machines[m_idx]:
                    if inspection_restricts[m_idx]:
                        # Inspection 강제 상황: Action 6 (Inspection), 내부 룰 2 (FIFO)
                        mc_action = 6
                        mc_insp_action = 2
                    else:
                        # 일반 상황: Action 2 (FIFO)
                        mc_action = 2
                        mc_insp_action = -1
                
                actions.extend([mc_action, mc_insp_action])
            
            
            # 3. Step 실행
            next_state, idle_times, blocked_times, done, next_insp_restricts, next_machines = env.step(action_list=actions)
            
            for idx in range(branch_num):
                total_idle_times[idx] += idle_times[idx]
                total_blocked_times[idx] += blocked_times[idx]

            current_buffer_lengths = [len(buffer) for buffer in env.buffers]
            for m_idx in range(1, len(current_buffer_lengths)):
                length = current_buffer_lengths[m_idx]
                if length > env.buffer_limit:
                    raise Exception(f"🚨 Buffer Capacity Exceeded! "
                                    f"Machine {m_idx+1}의 버퍼 개수가 {length}개로 제한({env.buffer_limit}개)을 초과했습니다.")
            
            # 5. 상태 업데이트
            available_machines = next_machines
            inspection_restricts = next_insp_restricts
            current_state = next_state
        
        makespan_list.append(env.makespan)
        
        # Monitoring
        if (i + 1) % 10 == 0:
            print(f"[Progress] Episode {i+1:3d}/{episode_num} Completed. (Makespan: {env.makespan})")

    env.close()
    
    # 6. Results
    average_makespan = sum(makespan_list) / episode_num
    avg_idle_times = [round(t / episode_num, 2) for t in total_idle_times]
    avg_blocked_times = [round(t / episode_num, 2) for t in total_blocked_times]
    
    print(f"\n{'=' * 40}")
    print(f"<<<<<<최종 테스트 결과 요약>>>>>>")
    print(f"{'=' * 40}")
    print(f" - 총 테스트 에피소드   : {episode_num} 회")
    print(f" - 작업 수 (Episode Len) : {temp_episode_len_list[0]} Jobs")
    print(f" - 버퍼 최대 용량 (Limit) : {BUFFER_LIMIT}")
    print(f" - 평균 소요 스텝 수     : {step_counter / episode_num:.2f} Steps")
    print(f" - 평균 Makespan        : {average_makespan:.2f}")
    print(f" - 평균 Idle Time       : {avg_idle_times}")
    print(f" - 평균 Blocked Time    : {avg_blocked_times}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()