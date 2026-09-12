' Launches run_collector.cmd without a console window and waits for it (so Task Scheduler sees the real process).
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "D:\kalshi\codes\dashboard"
sh.Run "cmd.exe /c ""D:\kalshi\codes\dashboard\run_collector.cmd""", 0, True
