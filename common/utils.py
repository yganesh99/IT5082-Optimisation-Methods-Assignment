def slot_to_time(slot_index, start_hour=10):
    # Assuming 15-minute slots
    total_minutes = slot_index * 15
    hour = start_hour + (total_minutes // 60)
    minutes = total_minutes % 60
    return f"{hour:02d}:{minutes:02d}"