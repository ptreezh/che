"""
Soul 自主进化引擎 v2.0
永不停止的持续进化机制

功能：
1. 持久化任务队列
2. 自动任务执行
3. 跨会话状态恢复
4. 检查点管理
5. 错误恢复与重试
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# 配置路径
SOUL_DIR = Path(__file__).parent.parent / ".soul"
EVOLUTION_FILE = SOUL_DIR / "evolution_state.json"
CHECKPOINT_DIR = SOUL_DIR / "checkpoints"
LOG_FILE = SOUL_DIR / "evolution.log"


class EvolutionEngine:
    """自主进化引擎"""
    
    def __init__(self):
        self.state = self._load_state()
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """确保必要目录存在"""
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_state(self) -> Dict:
        """加载进化状态"""
        if EVOLUTION_FILE.exists():
            with open(EVOLUTION_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self._create_initial_state()
    
    def _create_initial_state(self) -> Dict:
        """创建初始状态"""
        return {
            "version": "2.0.0",
            "created": datetime.now().isoformat(),
            "evolution_state": {"current_phase": "init", "completed_cycles": 0},
            "task_queue": [],
            "metrics": {},
            "checkpoints": {"last_checkpoint": datetime.now().isoformat()}
        }
    
    def save_state(self):
        """保存当前状态"""
        self.state["evolution_state"]["last_update"] = datetime.now().isoformat()
        with open(EVOLUTION_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)
        self._log("状态已保存")
    
    def _log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        print(log_entry.strip())
    
    def get_next_task(self) -> Optional[Dict]:
        """获取下一个待执行任务"""
        for task in self.state.get("task_queue", []):
            if task["status"] == "pending":
                # 检查依赖
                deps = task.get("dependencies", [])
                all_deps_done = all(
                    self._is_task_completed(dep_id) 
                    for dep_id in deps
                )
                if all_deps_done:
                    return task
        return None
    
    def _is_task_completed(self, task_id: str) -> bool:
        """检查任务是否完成"""
        for task in self.state.get("task_queue", []):
            if task["id"] == task_id:
                return task["status"] == "completed"
        return True  # 不存在的任务视为已完成
    
    def start_task(self, task: Dict):
        """开始执行任务"""
        task["status"] = "in_progress"
        task["started_at"] = datetime.now().isoformat()
        self.save_state()
        self._log(f"开始任务: {task['id']} - {task['task']}")
    
    def complete_task(self, task_id: str, result: Any = None):
        """完成任务"""
        for task in self.state["task_queue"]:
            if task["id"] == task_id:
                task["status"] = "completed"
                task["completed_at"] = datetime.now().isoformat()
                if result:
                    task["result"] = result
                self._log(f"完成任务: {task_id}")
        self.save_state()
        self._generate_follow_up_tasks(task_id)
    
    def fail_task(self, task_id: str, error: str):
        """任务失败"""
        for task in self.state["task_queue"]:
            if task["id"] == task_id:
                task["retry_count"] = task.get("retry_count", 0) + 1
                if task["retry_count"] >= task.get("max_retries", 3):
                    task["status"] = "failed"
                    task["error"] = error
                else:
                    task["status"] = "pending"  # 重试
                self._log(f"任务失败: {task_id} - {error}")
        self.save_state()
    
    def _generate_follow_up_tasks(self, completed_task_id: str):
        """生成后续任务"""
        # 基于完成的任务生成新任务
        rules = self.state.get("self_evolution_rules", {})
        if rules.get("auto_generate_next_tasks", True):
            # 检查是否需要生成新任务
            pending_count = sum(
                1 for t in self.state["task_queue"] 
                if t["status"] == "pending"
            )
            if pending_count < 3:
                self._auto_generate_tasks()
    
    def _auto_generate_tasks(self):
        """自动生成任务"""
        phase = self.state["evolution_state"].get("current_phase", "")
        
        new_tasks = []
        if phase == "phase_3_experimentation":
            new_tasks = [
                {
                    "id": f"auto_exp_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "type": "experiment",
                    "task": "运行额外验证实验",
                    "priority": "medium",
                    "status": "pending",
                    "created": datetime.now().isoformat(),
                    "dependencies": [],
                    "retry_count": 0,
                    "max_retries": 3
                }
            ]
        elif phase == "phase_4_publication":
            new_tasks = [
                {
                    "id": f"auto_pub_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "type": "publication",
                    "task": "完善论文细节",
                    "priority": "medium",
                    "status": "pending",
                    "created": datetime.now().isoformat(),
                    "dependencies": [],
                    "retry_count": 0,
                    "max_retries": 3
                }
            ]
        
        self.state["task_queue"].extend(new_tasks)
        if new_tasks:
            self._log(f"自动生成 {len(new_tasks)} 个新任务")
        self.save_state()
    
    def create_checkpoint(self):
        """创建检查点"""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "state": self.state.copy(),
            "summary": {
                "pending_tasks": sum(1 for t in self.state["task_queue"] if t["status"] == "pending"),
                "completed_tasks": sum(1 for t in self.state["task_queue"] if t["status"] == "completed"),
                "failed_tasks": sum(1 for t in self.state["task_queue"] if t["status"] == "failed")
            }
        }
        
        checkpoint_file = CHECKPOINT_DIR / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        self.state["checkpoints"]["last_checkpoint"] = datetime.now().isoformat()
        self.save_state()
        self._log(f"检查点已创建: {checkpoint_file.name}")
    
    def get_status_report(self) -> str:
        """获取状态报告"""
        lines = [
            "=" * 60,
            "SOUL 自主进化引擎状态报告",
            "=" * 60,
            f"当前阶段: {self.state['evolution_state'].get('current_phase', 'unknown')}",
            f"完成周期: {self.state['evolution_state'].get('completed_cycles', 0)}",
            f"最后更新: {self.state['evolution_state'].get('last_update', 'never')}",
            "",
            "【任务队列状态】",
        ]
        
        status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
        for task in self.state.get("task_queue", []):
            status = task.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        lines.extend([
            f"  待处理: {status_counts.get('pending', 0)}",
            f"  进行中: {status_counts.get('in_progress', 0)}",
            f"  已完成: {status_counts.get('completed', 0)}",
            f"  已失败: {status_counts.get('failed', 0)}",
            "",
            "【关键指标】",
        ])
        
        metrics = self.state.get("metrics", {})
        for key, value in metrics.items():
            lines.append(f"  {key}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def advance_phase(self):
        """推进到下一阶段"""
        phases = ["phase_1_foundation", "phase_2_validation", "phase_3_experimentation", "phase_4_publication"]
        current = self.state["evolution_state"].get("current_phase", phases[0])
        
        if current in phases:
            current_idx = phases.index(current)
            if current_idx < len(phases) - 1:
                next_phase = phases[current_idx + 1]
                self.state["evolution_state"]["current_phase"] = next_phase
                self.state["evolution_state"]["completed_cycles"] = self.state["evolution_state"].get("completed_cycles", 0) + 1
                self._log(f"阶段推进: {current} -> {next_phase}")
        
        self.save_state()


def main():
    """主函数 - CLI入口"""
    engine = EvolutionEngine()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "status":
            print(engine.get_status_report())
        
        elif command == "next":
            task = engine.get_next_task()
            if task:
                print(f"下一个任务: {task['id']} - {task['task']}")
            else:
                print("没有待处理任务")
        
        elif command == "checkpoint":
            engine.create_checkpoint()
            print("检查点已创建")
        
        elif command == "advance":
            engine.advance_phase()
            print("阶段已推进")
        
        else:
            print(f"未知命令: {command}")
            print("可用命令: status, next, checkpoint, advance")
    else:
        # 默认显示状态
        print(engine.get_status_report())


if __name__ == "__main__":
    main()
