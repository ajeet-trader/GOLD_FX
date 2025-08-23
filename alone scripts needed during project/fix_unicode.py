#!/usr/bin/env python3
"""
Fix Unicode characters in Phase 1 integration file
"""

# Read the file
with open('src/phase_1_core_integration.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define all replacements
replacements = {
    # Unicode escape sequences
    '\\u2705': 'OK -',
    '\\u274c': 'ERROR -',
    '\\U0001f4cb': 'Step',
    '\\U0001f3af': '',
    
    # Direct Unicode characters
    '\u2705': 'OK -',  # ✅
    '\u274c': 'ERROR -',  # ❌
    '\U0001f4cb': 'Step',  # 📋
    '\U0001f3af': '',  # 🎯
    '\U0001f389': '',  # 🎉
    '\U0001f4ca': 'Status:',  # 📊
    '\U0001f680': '',  # 🚀
    '\u26a0': 'WARNING -',  # ⚠️
    '\U0001f6d1': 'STOP -',  # 🛑
    '\U0001f4e1': '',  # 📡
    '\U0001f4a1': '',  # 💡
    '\U0001f527': '',  # 🔧
    '\u23f3': 'PENDING -',  # ⏳
    '\U0001f31f': '',  # 🌟
    '\u2b50': '',  # ⭐
    
    # Common emojis that might appear
    '✅': 'OK -',
    '❌': 'ERROR -',
    '📋': 'Step',
    '🎯': '',
    '🎉': '',
    '📊': 'Status:',
    '🚀': '',
    '⚠️': 'WARNING -',
    '🛑': 'STOP -',
    '📡': '',
    '💡': '',
    '🔧': '',
    '⏳': 'PENDING -',
    '🌟': '',
    '⭐': '',
}

# Apply all replacements
for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
with open('src/phase_1_core_integration.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("All Unicode characters have been replaced with ASCII equivalents")
