import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from npfsp_env.limited_env import Process
import matplotlib.lines as mlines

def plot_gantt_chart(env: Process, target_size: str, instance_idx: int = 0):
    """
    - A-BD3QN이 의사결정을 내린 환경의 기록을 바탕으로 기계 및 검사 단계의 병목/유휴/작업 타임라인을 시각화 한다. 
    - 에이전트가 자발적으로 검사 (Bypass) 결정을 내린 시점을 강조한다. 
    """
    if not hasattr(env, 'gantt_timeline') or not env.gantt_timeline:
        print("❌: 그려낼 간트 차트 타임라인 로그가 존재하지 않는다.")
        return 
    
    num_machines = env.num_machines
    fig, ax = plt.subplots(figsize=(12,6))

    color_map = {
        "Processing": "#4EA8DE", 
        "Inspection": "#70E000", 
        "Blocked": "#D90429",    
        "Idle": "#F8F9FA"         
    }

    # ==============================================================
    # 같은 job들의 block은 하나의 block으로 뭉쳐보이게끔 하기 위한 data merge과정.
    # ==============================================================
    merged_machines = {m: [] for m in range(num_machines)}
    merged_insps = {m: [] for m in range(num_machines - 1)}

    for snapshot in env.gantt_timeline:
        t_start = snapshot["time_start"]
        t_end = snapshot["time_end"]

        # 메인 기계 병합
        for m in range(num_machines):
            m_data = snapshot["machines"][m]
            if merged_machines[m] and merged_machines[m][-1]["status"] == m_data["status"] and merged_machines[m][-1]["job_num"] == m_data["job_num"]:
                merged_machines[m][-1]["end"] = t_end
            else:
                merged_machines[m].append({"start": t_start, "end": t_end, "status": m_data["status"], "job_num": m_data["job_num"]})

        # 검사 기계 병합
        for m in range(num_machines - 1):
            if m < len(snapshot["insps"]):
                i_data = snapshot["insps"][m]
                if merged_insps[m] and merged_insps[m][-1]["status"] == i_data["status"] and merged_insps[m][-1]["job_num"] == i_data["job_num"]:
                    merged_insps[m][-1]["end"] = t_end
                else:
                    merged_insps[m].append({"start": t_start, "end": t_end, "status": i_data["status"], "job_num": i_data["job_num"]})

    # ==============================================================
    # 1. 병합된 타임라인 데이터를 순회하며 Bar 생성
    # ==============================================================
    
    # main machine들 그리기
    for m in range(num_machines):
        for block in merged_machines[m]:
            if block["status"] != "Idle":
                t_start = block["start"]
                t_dur = block["end"] - block["start"]
                job_label = f"J{block['job_num']}" if block["job_num"] != -1 else ""

                ax.broken_barh([(t_start, t_dur)], (m * 2 + 0.6, 0.8), facecolors=color_map[block["status"]], edgecolor='black', linewidth=0.5)
                if t_dur > 5 and job_label: 
                    ax.text(t_start + t_dur/2, m * 2 + 1.0, job_label, ha='center', va='center', fontsize=8, color='white', weight='bold')

    # Inspection machine들 그리기
    for m in range(num_machines - 1):
        for block in merged_insps[m]:
            if block["status"] != "Idle":
                t_start = block["start"]
                t_dur = block["end"] - block["start"]
                job_label = f"J{block['job_num']}(I)" if block["job_num"] != -1 else ""

                ax.broken_barh([(t_start, t_dur)], (m * 2 + 1.4, 0.4), facecolors=color_map[block["status"]], edgecolor='black', linewidth=0.3, linestyle='--')

   # ==============================================================
    # 2. Main logic: PI(자발적 검사) 발생 시 해당 기계에만 짧은 점선 표시!
    # ==============================================================
    for hist in env.gantt_history:
        actions = hist["actions"]
        restricts = hist["restricts"]
        t_action = hist["time"]
        
        for m in range(num_machines - 1):
            m_act = actions[2 * m]
            if m_act == 6 and not restricts[m]:
                # 💡 axvline(전체 관통선) 대신 vlines(부분 선) 사용!
                # m * 2 위치가 해당 기계의 y축 좌표이므로, 그 기계의 블록 높이에만 선을 긋습니다.
                ax.vlines(x=t_action, ymin=m * 2 + 0.4, ymax=m * 2 + 1.9, 
                          colors='#CC0066', linestyles=':', linewidth=2, alpha=0.8)
    
    # ==============================================================
    # 3. CMAX (Makespan) 종료 지점 표시 
    # ==============================================================
    # 안전하게 10번째 기계(마지막 기계)의 가장 마지막 블록 끝나는 시간을 추출합니다.
    cmax_val = env.makespan 
    
    # 그래프 전체를 가로지르는 굵은 검은색 실선 생성
    ax.axvline(x=cmax_val, color='black', linestyle='-', linewidth=2.5, zorder=5)
    
    # x축 하단에 박스 형태로 CMAX 수치 표시 (눈에 확 띄게 처리)
    # transform=ax.get_xaxis_transform()을 쓰면 y=0이 X축 라인이 됩니다.
    ax.text(cmax_val, -0.02, f"Cmax: {cmax_val}", transform=ax.get_xaxis_transform(),
            color='black', ha='center', va='top', fontsize=11, weight='bold',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # ==============================================================
    # 4. 그래프 디자인 및 레이아웃 정비 
    # ==============================================================
    ax.set_title(f"Gantt Chart (m=6, n=15)", fontsize=14, pad=35)
    ax.set_xlabel("Time ($t$)", fontsize=12)

    y_ticks = [m * 2 + 1.0 for m in range(num_machines)]
    y_labels = [f"Machine {m+1}" for m in range(num_machines)]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11)
    
    ax.set_ylim(0, num_machines * 2 + 0.5)
    ax.set_xlim(0, 2000)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 💡 범례에 Makespan 표시선 설명 추가
    legend_elements = [
        mpatches.Patch(color=color_map["Processing"], label='Main Processing'),
        mpatches.Patch(color=color_map["Inspection"], label='Inspection Stage'),
        mpatches.Patch(color=color_map["Blocked"], label='Blocked Time (Buffer Full)'),
        mlines.Line2D([], [], color='#CC0066', linestyle=':', linewidth=2, label='Proactive Inspection (Action=6)'),
        mlines.Line2D([], [], color='black', linestyle='-', linewidth=2.5, label='Makespan ($C_{max}$)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0))

    plt.tight_layout()
    plt.show()