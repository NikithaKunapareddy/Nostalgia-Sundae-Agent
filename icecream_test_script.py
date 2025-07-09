
# Focused memory recall test: fact → unrelated messages → trigger recall
import requests
import time

API_URL = "http://localhost:5000/ask"
USER_ID = "icecream_tester"

# Helper to send a message to the bot
def send_message(message):
    payload = {"message": message}
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("response", "[No response]")
    except Exception as e:
        return f"[Error: {e}]"

def main():
    print("--- Ice Cream Memory Recall Test ---\n")

    # Step 1: Share related user facts

    fact2 = "I like eating ice cream when I'm anxious."
    print(f"User: {fact2}")
    print("Bot:", send_message(fact2))
    time.sleep(1)

    # Step 2: Distractor/unrelated messages
    distractors = [
        "I had a really productive day at work!",
        "I'm planning a vacation soon.",
        "I tried a new recipe for dinner yesterday.",
        "The weather is beautiful outside.",
        "I'm thinking about starting a new hobby.",
        "I watched a great movie last night.",
    ]
    for msg in distractors:
        print(f"User: {msg}")
        print("Bot:", send_message(msg))
        print("\n" + "-"*40 + "\n")  # Adds a visual gap after every conversation
        time.sleep(0.5)

    # Step 3: Trigger recall
    trigger = "I'm anxious."
    print(f"\nUser: {trigger}")
    print("Bot:", send_message(trigger))

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    main()
