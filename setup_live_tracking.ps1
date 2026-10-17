# setup_live_tracking.ps1 — Schedule the daily live-tracking run (run_live_tracking.py) in Windows Task Scheduler.
#
# Runs Monday to Friday at 18:30 local time, after NSE closes at 15:30 and Yahoo has the day's prices.
# If the PC is off or asleep at 18:30, the run starts as soon as it is available again.
#
#   Turn on:   powershell -ExecutionPolicy Bypass -File setup_live_tracking.ps1
#   Turn off:  Unregister-ScheduledTask -TaskName "AlphaEdge live tracking" -Confirm:$false
#   Run now:   Start-ScheduledTask -TaskName "AlphaEdge live tracking"
#   Status:    Get-ScheduledTaskInfo -TaskName "AlphaEdge live tracking"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$script = Join-Path $root 'run_live_tracking.py'
$taskName = 'AlphaEdge live tracking'

$action = New-ScheduledTaskAction -Execute $python -Argument "`"$script`"" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At 6:30PM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force `
    -Description 'AlphaEdge: fetch NSE prices, forecast with the final model and confidence flag, score past forecasts. Logs in logs\.' | Out-Null

$info = Get-ScheduledTask -TaskName $taskName | Get-ScheduledTaskInfo
Write-Output "Scheduled '$taskName': weekdays at 18:30. Next run: $($info.NextRunTime)"
