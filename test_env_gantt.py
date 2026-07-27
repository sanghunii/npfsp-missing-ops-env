from typing import List

from npfsp_env import limited_env
from npfsp_env.modules.gantt_sup import plot_gantt_chart

def main():
    episode_len = 15

    # ENV PARAMETERS
    MC_NUM = 6
    PROCESS_RANGE = [(20, 90)] * MC_NUM
    BUFFER_LIMIT = 3
    
    print(f"{'=' * 50}")
    print("🚀 NPFSP Environment Single Episode & Gantt Plotting...")
    print(f"{'=' * 50}\n")
    
    # 2. limited_env.Process 로 환경 초기화
    env = limited_env.Process(
        num_machines=MC_NUM, 
        process_range=PROCESS_RANGE, 
        episode_len_list=[episode_len], 
        buffer_limit=BUFFER_LIMIT
    )
    
    step_counter = 0
    total_idle_times = [0] * (MC_NUM - 1)
    total_blocked_times = [0] * (MC_NUM - 1)
    
    # 3. 환경 리셋 (Gantt 장부 초기화)
    current_state, done, inspection_restricts, available_machines = env.reset()
    branch_num = MC_NUM - 1 

    print("⏳ Running Simulation...")

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
        
        # 4. Step 실행 (Gantt log 기록)
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
    
    # 6. 시뮬레이션 종료 후 Gantt Chart 바로 출력
    print("🏁 Simulation complete! Plotting Gantt chart...\n")
    target_size_str = f"size{episode_len}"
    plot_gantt_chart(env=env, target_size=target_size_str, instance_idx=0)

    env.close()
    
    # 7. 단일 에피소드 결과 요약
    print(f"{'=' * 40}")
    print(f"<<<<<< 단일 테스트 결과 요약 >>>>>>")
    print(f"{'=' * 40}")
    print(f" - 작업 수 (Episode Len)  : {episode_len} Jobs")
    print(f" - 버퍼 최대 용량 (Limit) : {BUFFER_LIMIT}")
    print(f" - 총 소요 스텝 수        : {step_counter} Steps")
    print(f" - Makespan             : {env.makespan}")
    print(f" - 누적 Idle Time       : {total_idle_times}")
    print(f" - 누적 Blocked Time    : {total_blocked_times}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()