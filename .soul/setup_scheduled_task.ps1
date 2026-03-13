# SOUL 自主进化 - Windows计划任务配置
# 运行方式：以管理员身份执行此脚本

$TaskName = "SOUL_Evolution_CHE_Project"
$ScriptPath = "D:\AIDevelop\che_project\.soul\auto_execute.py"
$PythonPath = (Get-Command python).Source
$WorkingDir = "D:\AIDevelop\che_project\.soul"

# 检查任务是否已存在
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($ExistingTask) {
    Write-Host "任务已存在，更新配置..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# 创建触发器：每小时执行一次
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)

# 创建动作
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $ScriptPath -WorkingDirectory $WorkingDir

# 创建设置
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -AllowStartIfOnBatteries

# 注册任务（以当前用户身份运行）
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Settings $Settings -Description "SOUL自主进化引擎 - 每小时自动执行任务"

Write-Host "`n✅ 计划任务创建成功！"
Write-Host "任务名称: $TaskName"
Write-Host "执行间隔: 每小时"
Write-Host "脚本路径: $ScriptPath"
Write-Host "`n查看任务: Get-ScheduledTask -TaskName $TaskName"
Write-Host "手动执行: Start-ScheduledTask -TaskName $TaskName"
Write-Host "停止任务: Stop-ScheduledTask -TaskName $TaskName"
Write-Host "删除任务: Unregister-ScheduledTask -TaskName $TaskName"
