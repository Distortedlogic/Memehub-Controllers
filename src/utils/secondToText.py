def secondsToText(secs: int) -> str:
    days = int(secs // 86400)
    hours = int((secs - days * 86400) // 3600)
    minutes = int((secs - days * 86400 - hours * 3600) // 60)
    seconds = int(secs - days * 86400 - hours * 3600 - minutes * 60)
    result = (
        ("{0} day{1}, ".format(days, "s" if days != 1 else "") if days else "")
        + ("{0} hour{1}, ".format(hours, "s" if hours != 1 else "") if hours else "")
        + (
            "{0} minute{1}, ".format(minutes, "s" if minutes != 1 else "")
            if minutes
            else ""
        )
        + (
            "{0} second{1}, ".format(seconds, "s" if seconds != 1 else "")
            if seconds
            else ""
        )
    )
    return result
