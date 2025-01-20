import random
import datetime
import customtkinter as ctk
from tkcalendar import Calendar
import sqlite3
import pyttsx3
import threading
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import pyaudio
from vosk import Model, KaldiRecognizer
import sys
import os
import re
import difflib
import dateparser
from word2number import w2n
import tkinter as tk
import numpy as np

# -----------------------
# Text-to-Speech Setup
# -----------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)     # speaking speed
engine.setProperty('volume', 1.0)   # full volume

def speak(text):
    """Safely speak the given text."""
    try:
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        pass

# -----------------------
# Database Setup
# -----------------------
thread_local = threading.local()

def create_connection():
    """Create a thread-local connection to the SQLite database."""
    thread_local.conn = sqlite3.connect('salon_schedule.db')
    thread_local.cursor = thread_local.conn.cursor()

def get_cursor():
    """Return the thread-local cursor."""
    return thread_local.cursor

def get_connection():
    """Return the thread-local connection."""
    return thread_local.conn

create_connection()

# Create our tables if they don't exist:
get_cursor().execute('''
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    phone TEXT NOT NULL
)
''')
get_cursor().execute('''
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    service TEXT NOT NULL,
    worker TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (id)
)
''')
get_connection().commit()

# For record-keeping
conversation_history = []
def log_conversation(user_input, app_response):
    conversation_history.append({"user": user_input, "response": app_response})
    print(f"User: {user_input}, App: {app_response}")

# -----------------------
# Speech Recognition Setup
# (if you actually need voice in real usage)
# -----------------------
model_path = r"C:\Users\ameer\Downloads\vosk-model-en-us-0.22\vosk-model-en-us-0.22"
if not os.path.exists(model_path):
    print("Please provide the correct path to your Vosk model.")
    # sys.exit(1)  # comment out if you don't need voice
vosk_model = Model(model_path)

def recognize_speech():
    """Continuously listen to the mic and return recognized text once detected."""
    recognizer = KaldiRecognizer(vosk_model, 16000)
    mic = pyaudio.PyAudio()
    try:
        stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000,
                          input=True, frames_per_buffer=8192)
        stream.start_stream()
        print("Listening...")
        text = ""
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                result_json = json.loads(result)
                text = result_json.get('text', '')
                break
    finally:
        stream.stop_stream()
        stream.close()
        mic.terminate()
    return text

# -----------------------
# Conversation State
# -----------------------
class ConversationState:
    """Track what the user is trying to do (intent) and data collected."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.intent = None
        self.data = {}
        self.step = None

conversation_state = ConversationState()

# -----------------------
# Services & Workers
# -----------------------
SERVICES = {
    "Haircut":  {"duration": 30, "price": 25},
    "Hair Color": {"duration": 60, "price": 60},
    "Manicure": {"duration": 45, "price": 20},
    "Pedicure": {"duration": 60, "price": 25},
    "Styling":  {"duration": 45, "price": 30},
    "Shave":    {"duration": 20, "price": 15},
    "Facial":   {"duration": 30, "price": 40}
}

SERVICE_SYNONYMS = {
    "haircut":   ["cut", "trim", "hair cut"],
    "hair color":["coloring", "dye", "hair dye"],
    "manicure":  ["nail", "nails", "mani"],
    "pedicure":  ["pedi", "feet"],
    "styling":   ["style", "hairstyle"],
    "shave":     ["beard trim", "beard"],
    "facial":    ["face treatment", "skin care"]
}

WORKERS = ["Alice", "Bob", "Charlie", "David"]

# -----------------------
# Helpers for NLP
# -----------------------
def match_service(user_input):
    """Use synonyms to figure out which service the user wants."""
    user_input = user_input.lower()
    for service, synonyms in SERVICE_SYNONYMS.items():
        # also include the exact service word itself
        synonyms_with_service = synonyms + [service]
        for synonym in synonyms_with_service:
            if synonym in user_input:
                return service.title()
    return None

def fuzzy_match(word, possibilities):
    """Close match approach using difflib."""
    matches = difflib.get_close_matches(word, possibilities, n=1, cutoff=0.7)
    return matches[0] if matches else None

def extract_date(user_input):
    """Use dateparser to attempt parsing a date from user input."""
    date = dateparser.parse(user_input, settings={'PREFER_DATES_FROM': 'future'})
    return date.date() if date else None

def extract_time(user_input):
    """Attempt to parse a time from user input."""
    time = dateparser.parse(user_input)
    return time.time() if time else None

def get_available_workers(date, time, service_duration):
    """
    Return which workers are free at the given date/time for the
    specified duration. Checks the DB for existing appointments.
    """
    available_workers = WORKERS.copy()
    try:
        appointment_start = datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return available_workers

    appointment_end = appointment_start + datetime.timedelta(minutes=service_duration)
    for worker in WORKERS:
        get_cursor().execute('''
            SELECT time, service
            FROM appointments
            WHERE worker = ? AND date = ?
        ''', (worker, date))
        existing_appointments = get_cursor().fetchall()
        for existing_time_str, existing_service in existing_appointments:
            try:
                existing_start = datetime.datetime.strptime(f"{date} {existing_time_str}", "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            existing_duration = SERVICES.get(existing_service, {"duration": 30})["duration"]
            existing_end = existing_start + datetime.timedelta(minutes=existing_duration)
            # If there's overlap for that worker, remove them from availability
            if (appointment_start < existing_end) and (appointment_end > existing_start):
                if worker in available_workers:
                    available_workers.remove(worker)
                break
    return available_workers

# -----------------------
# Conflict / Idle / Revenue
# -----------------------
def count_conflicts(schedule):
    """
    Count how many appointments in 'schedule' overlap in time
    *for the same worker* OR *the same customer*.
    """
    conflicts = 0

    # For convenience, group appointments by worker, but also track by customer.
    # We'll check overlap within each grouping.
    worker_schedules = {w: [] for w in WORKERS}
    customer_schedules = {}

    def parse_dt(appt):
        return datetime.datetime.strptime(f"{appt['date']} {appt['time']}", "%Y-%m-%d %H:%M")

    for appt in schedule:
        worker = appt['worker']
        customer = appt.get('name', 'Unknown')  # or phone, if you prefer unique
        svc = appt['service']
        duration = SERVICES.get(svc, {"duration": 30})["duration"]
        start_time = parse_dt(appt)
        end_time = start_time + datetime.timedelta(minutes=duration)

        # Check conflict with same worker
        for w_appt in worker_schedules[worker]:
            w_duration = SERVICES.get(w_appt['service'], {"duration": 30})["duration"]
            w_start = parse_dt(w_appt)
            w_end = w_start + datetime.timedelta(minutes=w_duration)
            # If there's time overlap for the same worker
            if (start_time < w_end) and (end_time > w_start):
                conflicts += 1

        # Insert into worker structure
        worker_schedules[worker].append(appt)

        # Also track by customer
        if customer not in customer_schedules:
            customer_schedules[customer] = []
        for c_appt in customer_schedules[customer]:
            c_duration = SERVICES.get(c_appt['service'], {"duration": 30})["duration"]
            c_start = parse_dt(c_appt)
            c_end = c_start + datetime.timedelta(minutes=c_duration)
            # If there's time overlap for the same customer
            if (start_time < c_end) and (end_time > c_start):
                conflicts += 1

        customer_schedules[customer].append(appt)

    return conflicts

def calculate_total_idle_time(schedule):
    """
    Sum up the idle time in minutes across all workers, based on gaps
    between appointments in chronological order.
    """
    idle_time = 0
    worker_schedules = {w: [] for w in WORKERS}

    def parse_dt(appt):
        return datetime.datetime.strptime(f"{appt['date']} {appt['time']}", "%Y-%m-%d %H:%M")

    for appt in schedule:
        worker_schedules[appt['worker']].append(appt)

    for w in WORKERS:
        worker_schedules[w].sort(key=lambda a: parse_dt(a))
        for i in range(1, len(worker_schedules[w])):
            prev_appt = worker_schedules[w][i-1]
            curr_appt = worker_schedules[w][i]
            prev_dur = SERVICES.get(prev_appt['service'], {"duration": 30})["duration"]
            prev_end = parse_dt(prev_appt) + datetime.timedelta(minutes=prev_dur)
            curr_start = parse_dt(curr_appt)
            gap = (curr_start - prev_end).total_seconds() / 60.0
            if gap > 0:
                idle_time += gap
    return idle_time

def calculate_total_revenue(schedule):
    """
    Sum up the 'price' for each appointment in 'schedule' based on the service.
    """
    total = 0
    for appt in schedule:
        total += SERVICES.get(appt['service'], {"price": 0})["price"]
    return total

def calculate_total_ratio(schedule):
    """
    Sum up price/duration for each appointment, used as part of the fitness.
    """
    total_ratio = 0
    for appt in schedule:
        svc = SERVICES.get(appt['service'], {"duration": 1, "price": 0})
        dur = max(svc["duration"], 1)
        total_ratio += (svc["price"] / dur)
    return total_ratio

# ----------------------
# Greedy Scheduling
# ----------------------
def run_greedy_schedule(appointments):
    """
    Sort by date/time, then schedule if the requested worker is free
    at that time. If worker is busy, skip. (Doesn't consider double-booking
    by the same customer though.)
    """
    def parse_dt(appt):
        dt_str = appt['date'] + " " + appt['time']
        return datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M")

    sorted_appts = sorted(appointments, key=parse_dt)
    schedule = []
    worker_schedules = {w: [] for w in WORKERS}

    for appt in sorted_appts:
        requested_worker = appt['worker']
        service_duration = SERVICES.get(appt['service'], {"duration": 30})["duration"]
        start_time = parse_dt(appt)
        end_time = start_time + datetime.timedelta(minutes=service_duration)

        conflict = False
        for existing_appt in worker_schedules[requested_worker]:
            existing_dur = SERVICES.get(existing_appt['service'], {"duration": 30})["duration"]
            existing_start = parse_dt(existing_appt)
            existing_end = existing_start + datetime.timedelta(minutes=existing_dur)
            # If overlap for that worker
            if (start_time < existing_end) and (end_time > existing_start):
                conflict = True
                break

        if not conflict:
            schedule.append(appt)
            worker_schedules[requested_worker].append(appt)

    return schedule

# ----------------------
# Genetic Algorithm
# ----------------------
def generate_initial_population(size, appointments):
    """
    Create 'size' random solutions by randomly assigning workers
    to each appointment (ignoring the user's requested worker).
    """
    population = []
    for _ in range(size):
        individual = []
        for appt in appointments:
            appt_copy = appt.copy()
            appt_copy['worker'] = random.choice(WORKERS)
            individual.append(appt_copy)
        population.append(individual)
    return population

def repair_schedule(individual):
    """
    Fix any double-booking either by the same worker OR by the same customer.
    If conflict is for the worker, we try a different worker.
    If conflict is for the same customer, reassigning worker won't help—skip it.
    """
    repaired = []
    worker_schedules = {w: [] for w in WORKERS}
    customer_schedules = {}

    def parse_dt(appt):
        return datetime.datetime.strptime(appt['date']+" "+appt['time'], "%Y-%m-%d %H:%M")

    # Sort so we handle earlier appointments first
    sorted_individual = sorted(individual, key=parse_dt)

    for appt in sorted_individual:
        worker = appt['worker']
        customer = appt.get('name', 'Unknown')
        svc_dur = SERVICES.get(appt['service'], {"duration": 30})["duration"]
        start = parse_dt(appt)
        end = start + datetime.timedelta(minutes=svc_dur)

        conflict_found = False
        conflict_due_to_customer = False

        # Check existing for same worker
        for w_appt in worker_schedules[worker]:
            w_dur = SERVICES.get(w_appt['service'], {"duration": 30})["duration"]
            w_start = parse_dt(w_appt)
            w_end = w_start + datetime.timedelta(minutes=w_dur)
            if (start < w_end) and (end > w_start):
                # conflict with worker
                conflict_found = True
                break

        # Check existing for same customer
        if not conflict_found:
            if customer not in customer_schedules:
                customer_schedules[customer] = []
            for c_appt in customer_schedules[customer]:
                c_dur = SERVICES.get(c_appt['service'], {"duration": 30})["duration"]
                c_start = parse_dt(c_appt)
                c_end = c_start + datetime.timedelta(minutes=c_dur)
                if (start < c_end) and (end > c_start):
                    # conflict with same customer
                    conflict_found = True
                    conflict_due_to_customer = True
                    break

        if not conflict_found:
            # all good, add to schedule
            repaired.append(appt)
            worker_schedules[worker].append(appt)
            if customer not in customer_schedules:
                customer_schedules[customer] = []
            customer_schedules[customer].append(appt)

        else:
            # There's a conflict
            if conflict_due_to_customer:
                # Reassigning worker won't fix a same-customer conflict at the same time => skip
                continue
            else:
                # conflict due to the same worker => try reassigning to a free worker
                alternative_found = False
                for w in WORKERS:
                    # check if w is free at that time
                    # check worker conflict
                    w_conflict = False
                    for w_appt in worker_schedules[w]:
                        w_dur = SERVICES.get(w_appt['service'], {"duration": 30})["duration"]
                        w_start = parse_dt(w_appt)
                        w_end = w_start + datetime.timedelta(minutes=w_dur)
                        if (start < w_end) and (end > w_start):
                            w_conflict = True
                            break
                    # check same-customer conflict
                    if not w_conflict:
                        # also must check if the same customer is free
                        cust_conflict = False
                        if customer not in customer_schedules:
                            customer_schedules[customer] = []
                        for c_appt in customer_schedules[customer]:
                            c_dur = SERVICES.get(c_appt['service'], {"duration": 30})["duration"]
                            c_start = parse_dt(c_appt)
                            c_end = c_start + datetime.timedelta(minutes=c_dur)
                            if (start < c_end) and (end > c_start):
                                cust_conflict = True
                                break
                        if not cust_conflict:
                            # good, we can reassign
                            appt_copy = appt.copy()
                            appt_copy['worker'] = w
                            repaired.append(appt_copy)
                            worker_schedules[w].append(appt_copy)
                            customer_schedules[customer].append(appt_copy)
                            alternative_found = True
                            break
                # if we never found an alternative, we skip
                if not alternative_found:
                    pass

    return repaired

def fitness(individual):
    """
    Higher is better. We penalize conflicts heavily, penalize idle time,
    and reward high (price/duration).
    """
    conflict_count = count_conflicts(individual)
    idle_time = calculate_total_idle_time(individual)
    ratio_sum = calculate_total_ratio(individual)

    BIG_CONFLICT_PENALTY = 999
    IDLE_TIME_WEIGHT = 0.5
    RATIO_BONUS_WEIGHT = 5.0

    cost = (conflict_count * BIG_CONFLICT_PENALTY) \
           + (IDLE_TIME_WEIGHT * idle_time) \
           - (RATIO_BONUS_WEIGHT * ratio_sum)

    if cost < 0:
        cost = 0
    return 1 / (1 + cost)

def tournament_selection(population, fitnesses, k=3):
    """Pick k random solutions, return the best one."""
    selected = random.sample(list(zip(population, fitnesses)), k)
    return max(selected, key=lambda x: x[1])[0]

def selection(population, fitnesses, tournament_size=3):
    """Perform two tournament selections for parent1 and parent2."""
    parent1 = tournament_selection(population, fitnesses, k=tournament_size)
    parent2 = tournament_selection(population, fitnesses, k=tournament_size)
    return parent1, parent2

def crossover(parent1, parent2):
    """Simple single/multi‐point crossover of the schedule lists."""
    if len(parent1) < 2:
        return parent1[:], parent2[:]
    point1 = random.randint(1, len(parent1) - 1)
    point2 = random.randint(point1, len(parent1) - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2

def mutate(individual, mutation_rate):
    """With probability=mutation_rate, pick a random worker for each appointment."""
    for appt in individual:
        if random.random() < mutation_rate:
            appt['worker'] = random.choice(WORKERS)

def elitism(population, fitnesses, elite_size=10):
    """Keep the top N solutions from the population to carry over."""
    sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
    return [population[i] for i in sorted_indices[:elite_size]]

def local_search(schedule, max_iterations=10):
    """
    Try small random tweaks: reassign one appointment to another worker
    if it helps the fitness. We'll also re-check conflicts with `repair_schedule()`.
    """
    best_schedule = schedule
    best_fitness = fitness(schedule)

    for _ in range(max_iterations):
        if not best_schedule:
            break
        idx = random.randint(0, len(best_schedule) - 1)
        original_appt = best_schedule[idx]
        original_worker = original_appt['worker']

        # attempt to assign a new worker
        alternative_workers = [w for w in WORKERS if w != original_worker]
        random.shuffle(alternative_workers)

        improved = False
        for w in alternative_workers:
            new_appt = original_appt.copy()
            new_appt['worker'] = w
            candidate = best_schedule[:idx] + [new_appt] + best_schedule[idx+1:]
            candidate = repair_schedule(candidate)
            candidate_fitness = fitness(candidate)
            if candidate_fitness > best_fitness:
                best_schedule = candidate
                best_fitness = candidate_fitness
                improved = True
                break
        # if not improved, do nothing
    return best_schedule

def run_genetic_algorithm(appointments, generations=300, population_size=100, mutation_rate=0.4, attempts=1):
    """
    Main GA loop with multiple attempts if desired.
    We'll keep the best overall schedule found.
    """
    def ratio(appt):
        svc = SERVICES.get(appt['service'], {"duration": 30, "price": 0})
        dur = max(svc["duration"], 1)
        return svc["price"] / dur

    # Sort appointments by ratio=price/duration desc, so bigger payoff first
    sorted_appts = sorted(appointments, key=ratio, reverse=True)

    best_schedule_overall = None
    best_fitness_overall = -1

    for attempt in range(1, attempts+1):
        print(f"\nGA Attempt {attempt} starting...")

        population = generate_initial_population(population_size, sorted_appts)

        no_improvement_counter = 0
        max_stagnation = 100
        last_best_in_pop = -1

        for generation in range(generations):
            fitnesses = [fitness(ind) for ind in population]
            elites = elitism(population, fitnesses, elite_size=10)
            new_population = elites[:]

            best_in_pop = max(fitnesses)
            if best_in_pop > last_best_in_pop:
                last_best_in_pop = best_in_pop
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            if no_improvement_counter > max_stagnation:
                print("Early stopping due to stagnation.")
                break

            while len(new_population) < population_size:
                parent1, parent2 = selection(population, fitnesses, tournament_size=3)
                child1, child2 = crossover(parent1, parent2)
                mutate(child1, mutation_rate)
                mutate(child2, mutation_rate)
                # fix conflicts
                child1 = repair_schedule(child1)
                child2 = repair_schedule(child2)
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)

            population = new_population

        fitnesses = [fitness(ind) for ind in population]
        best_index = fitnesses.index(max(fitnesses))
        final_schedule = population[best_index]
        final_fitness = fitnesses[best_index]

        # local search
        improved_schedule = local_search(final_schedule, max_iterations=20)
        improved_fitness = fitness(improved_schedule)
        if improved_fitness > final_fitness:
            final_schedule = improved_schedule
            final_fitness = improved_fitness

        if final_fitness > best_fitness_overall:
            best_fitness_overall = final_fitness
            best_schedule_overall = final_schedule

    return best_schedule_overall

# ----------------------
# Booking & Bot Logic
# ----------------------
def confirm_booking():
    """
    Insert a new appointment in DB if the worker is free,
    otherwise skip. 
    """
    try:
        data = conversation_state.data
        customer_name = data.get("name")
        customer_phone = data.get("phone")
        service = data.get("service")
        date = data.get("date")
        time = data.get("time")
        service_duration = data.get("duration", 30)
        requested_worker_list = data.get("available_workers", WORKERS)

        worker = requested_worker_list[0] if requested_worker_list else random.choice(WORKERS)
        appointment_start = datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        appointment_end = appointment_start + datetime.timedelta(minutes=service_duration)

        # check if that worker is free
        get_cursor().execute('''
            SELECT time, service
            FROM appointments
            WHERE worker = ? AND date = ?
        ''', (worker, date))
        existing_appointments = get_cursor().fetchall()
        for existing_time_str, existing_service in existing_appointments:
            existing_start = datetime.datetime.strptime(f"{date} {existing_time_str}", "%Y-%m-%d %H:%M")
            existing_duration = SERVICES.get(existing_service, {"duration": 30})["duration"]
            existing_end = existing_start + datetime.timedelta(minutes=existing_duration)
            if (appointment_start < existing_end) and (appointment_end > existing_start):
                return "Requested worker is not free; skipping appointment."

        # Also check if *customer* is free
        get_cursor().execute('''
            SELECT a.time, a.service
            FROM appointments a
            JOIN customers c ON a.customer_id = c.id
            WHERE c.name = ? AND c.phone = ? AND a.date = ?
        ''', (customer_name, customer_phone, date))
        same_customer_appts = get_cursor().fetchall()
        for cust_time_str, cust_service in same_customer_appts:
            cust_start = datetime.datetime.strptime(f"{date} {cust_time_str}", "%Y-%m-%d %H:%M")
            cust_duration = SERVICES.get(cust_service, {"duration": 30})["duration"]
            cust_end = cust_start + datetime.timedelta(minutes=cust_duration)
            if (appointment_start < cust_end) and (appointment_end > cust_start):
                return f"It appears {customer_name} is already booked at this time. Skipping."

        # insert new customer if they don't exist
        get_cursor().execute("INSERT INTO customers (name, phone) VALUES (?, ?)", (customer_name, customer_phone))
        get_connection().commit()

        # retrieve customer id
        get_cursor().execute("SELECT id FROM customers WHERE name = ? AND phone = ?",
                             (customer_name, customer_phone))
        customer = get_cursor().fetchone()
        if not customer:
            return "An unexpected error occurred. Please try again."
        customer_id = customer[0]

        # Insert appointment
        get_cursor().execute('''
            INSERT INTO appointments (customer_id, date, time, service, worker)
            VALUES (?, ?, ?, ?, ?)
        ''', (customer_id, date, time, service, worker))
        get_connection().commit()

        confirmation = f"✅ Your {service} is scheduled for {date} at {time} with {worker}. Thank you, {customer_name}!"
        return confirmation

    except Exception as e:
        return f"Error booking appointment: {str(e)}"

def get_bot_response(user_input):
    """
    Decide how to respond to user text. We handle greetings, farewells,
    booking steps, etc.
    """
    ui_lower = user_input.lower()
    response = ""

    # Basic greetings
    if any(greet in ui_lower for greet in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        response = "Hello! My name is AI. How can I assist you today?"

    # Farewells
    elif any(farewell in ui_lower for farewell in ["bye", "goodbye", "exit", "see you", "later"]):
        response = "Goodbye! Feel free to reach out if you need anything else."
        conversation_state.reset()

    # Thanks
    elif any(thanks in ui_lower for thanks in ["thank you", "thanks", "appreciate"]):
        response = "You're welcome! Is there anything else I can assist you with?"

    # Booking logic
    elif any(word in ui_lower for word in ["book", "appointment", "schedule", "reserve", "make an appointment"]):
        conversation_state.intent = "booking"
        conversation_state.step = "service"
        response = "Sure! What service are you interested in?"

    elif any(word in ui_lower for word in ["services", "service", "offer", "do you have", "provide"]):
        svcs = ', '.join(SERVICES.keys())
        response = f"We offer: {svcs}. Which one would you like?"

    elif conversation_state.intent == "booking":
        # If we're in the middle of booking steps
        if conversation_state.step == "service":
            service = match_service(ui_lower)
            if service:
                conversation_state.data["service"] = service
                conversation_state.data["duration"] = SERVICES[service]["duration"]
                response = f"Great! When would you like your {service}?"
                conversation_state.step = "date"
            else:
                response = "I'm sorry, I couldn't detect the service. Could you please specify?"
        elif conversation_state.step == "date":
            appt_date = extract_date(ui_lower)
            if appt_date:
                conversation_state.data["date"] = appt_date.strftime("%Y-%m-%d")
                response = f"Got it. What time on {appt_date.strftime('%A, %B %d')}?"
                conversation_state.step = "time"
            else:
                response = "Could you please specify the date you'd like?"
        elif conversation_state.step == "time":
            appt_time = extract_time(ui_lower)
            if appt_time:
                time_str = appt_time.strftime("%H:%M")
                conversation_state.data["time"] = time_str
                svc_dur = conversation_state.data["duration"]
                # figure out which workers are free
                free_workers = get_available_workers(conversation_state.data["date"], time_str, svc_dur)
                conversation_state.data["available_workers"] = free_workers
                response = f"Available workers at that time: {', '.join(free_workers)}.\nYour name, please?"
                conversation_state.step = "name"
            else:
                response = "Please specify a valid time (e.g., 10:30 AM)."
        elif conversation_state.step == "name":
            conversation_state.data["name"] = user_input.title()
            response = "Thanks! Finally, may I have your phone number?"
            conversation_state.step = "phone"
        elif conversation_state.step == "phone":
            # parse phone
            try:
                numeric_input = ''.join(str(w2n.word_to_num(word)) for word in user_input.split())
            except ValueError:
                numeric_input = re.sub(r'\D', '', user_input)

            if len(numeric_input) >= 10:
                conversation_state.data["phone"] = numeric_input[-10:]
                # finalize booking
                response = confirm_booking()
                conversation_state.reset()
            else:
                response = "Please provide a phone number with at least 10 digits."
    else:
        response = "I'm sorry, I didn't quite catch that. Could you please rephrase?"

    return response

# ----------------------
# Misc. Print & Compare
# ----------------------
def sort_schedule_by_date_time(schedule):
    """Sort the schedule by date/time ascending."""
    def parse_dt(a):
        return datetime.datetime.strptime(f"{a['date']} {a['time']}", "%Y-%m-%d %H:%M")
    return sorted(schedule, key=parse_dt)

def print_worker_comparison(appointments, greedy_schedule, ga_schedule):
    """
    Show differences in worker assignments between Greedy & GA for each same date/time/service.
    """
    def index_schedule(schedule):
        d = {}
        for appt in schedule:
            key = (appt['date'], appt['time'], appt['service'])
            d[key] = appt['worker']
        return d

    g_index = index_schedule(greedy_schedule)
    ga_index = index_schedule(ga_schedule)

    print("\nDifferences in assignments (Greedy vs. GA):")
    for appt in appointments:
        key = (appt['date'], appt['time'], appt['service'])
        g_worker = g_index.get(key, "Skipped")
        ga_worker = ga_index.get(key, "Skipped")
        if g_worker != ga_worker:
            print(f" {key}: Greedy => {g_worker}, GA => {ga_worker}")

def load_data_from_file():
    """
    If the DB has zero appointments, load from appointments_data.json (if exists).
    """
    get_cursor().execute("SELECT COUNT(*) FROM appointments")
    count = get_cursor().fetchone()[0]
    if count == 0:
        if os.path.exists("appointments_data.json"):
            with open("appointments_data.json", "r") as f:
                data = json.load(f)
                for entry in data:
                    name = entry.get("name", "Unknown")
                    phone = entry.get("phone", "0000000000")
                    date = entry.get("date", "2024-12-20")
                    time = entry.get("time", "09:00")
                    service = entry.get("service", "Haircut")
                    worker = entry.get("worker", "Alice")

                    # insert new customer
                    get_cursor().execute("INSERT INTO customers (name, phone) VALUES (?,?)", (name, phone))
                    get_connection().commit()
                    get_cursor().execute("SELECT id FROM customers WHERE name=? AND phone=?", (name, phone))
                    customer = get_cursor().fetchone()
                    if customer:
                        customer_id = customer[0]
                        get_cursor().execute('''
                            INSERT INTO appointments (customer_id, date, time, service, worker)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (customer_id, date, time, service, worker))
                        get_connection().commit()
        else:
            print("No appointments_data.json file found to load. DB remains empty.")

# ----------------------
# Main App (GUI)
# ----------------------
class SalonSchedulerApp:
    """
    Main GUI application using customtkinter, plus your scheduling logic.
    """
    def __init__(self, master):
        self.master = master
        self.master.title("Salon Appointment Scheduler")
        self.master.geometry("1400x900")
        self.running = True
        self.setup_ui()
        self.run_and_compare_schedules()

    def setup_ui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Sidebar
        self.sidebar = ctk.CTkFrame(self.master, width=250, corner_radius=0, fg_color="#2e2e2e")
        self.sidebar.pack(side="left", fill="y")

        ctk.CTkLabel(self.sidebar, text="Salon Scheduler",
                     font=("Helvetica", 24, "bold"),
                     text_color="#ffffff").pack(pady=30)

        button_config = {
            "width": 220,
            "height": 50,
            "fg_color": "#3a3a3a",
            "hover_color": "#4a4a4a",
            "corner_radius": 10,
            "font": ("Helvetica", 14, "bold"),
            "text_color": "#ffffff",
            "border_width": 0
        }
        ctk.CTkButton(self.sidebar, text="📊 Dashboard", command=self.show_dashboard, **button_config).pack(pady=10, padx=15)
        ctk.CTkButton(self.sidebar, text="➕ New Appointment", command=self.show_appointment_form, **button_config).pack(pady=10, padx=15)
        ctk.CTkButton(self.sidebar, text="👁️ View Appointments", command=self.view_appointments, **button_config).pack(pady=10, padx=15)
        ctk.CTkButton(self.sidebar, text="🔍 Search Appointments", command=self.search_appointments, **button_config).pack(pady=10, padx=15)
        ctk.CTkButton(self.sidebar, text="🤖 Talk to AI", command=self.ai_conversation, **button_config).pack(pady=10, padx=15)

        # Main area to the right
        self.main_frame = ctk.CTkFrame(self.master, fg_color="#2e2e2e")
        self.main_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        self.status_label = ctk.CTkLabel(self.main_frame, text="Welcome to Salon Scheduler!",
                                         font=("Helvetica", 18),
                                         text_color="#ffffff")
        self.status_label.pack(pady=10)

        self.show_dashboard()

    def clear_main_frame(self):
        """Destroy all widgets in the main_frame to load something new."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_dashboard(self):
        """Show a simple dashboard with stats."""
        self.clear_main_frame()
        scrollable_frame = ctk.CTkScrollableFrame(self.main_frame, fg_color="#2e2e2e", scrollbar_button_color="#6e6e6e")
        scrollable_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(scrollable_frame, text="Dashboard",
                     font=("Helvetica", 28, "bold"),
                     text_color="#ffffff").pack(pady=20)

        appts_count = self.get_total_appointments_this_week()
        returning_cust = self.get_returning_customers()
        top_wrk = self.get_top_workers()
        top_svc = self.get_top_services()
        monthly_rev = self.get_monthly_revenue()

        ctk.CTkLabel(scrollable_frame,
                     text=f"Total Appointments This Week: {appts_count}",
                     font=("Helvetica", 18),
                     text_color="#ffffff").pack(pady=10)
        ctk.CTkLabel(scrollable_frame,
                     text=f"Returning Customers: {returning_cust}",
                     font=("Helvetica", 18),
                     text_color="#ffffff").pack(pady=10)

        worker_text = "Top Workers:\n"
        for w, c in top_wrk:
            worker_text += f"{w}: {c} appointments\n"
        ctk.CTkLabel(scrollable_frame, text=worker_text,
                     font=("Helvetica", 18),
                     text_color="#ffffff").pack(pady=10)

        service_text = "Top Services:\n"
        for s, c in top_svc:
            service_text += f"{s}: {c} appointments\n"
        ctk.CTkLabel(scrollable_frame, text=service_text,
                     font=("Helvetica", 18),
                     text_color="#ffffff").pack(pady=10)

        rev_text = "Monthly Revenue:\n"
        for month, rev in monthly_rev:
            rev_text += f"{month}: ${rev}\n"
        ctk.CTkLabel(scrollable_frame, text=rev_text,
                     font=("Helvetica", 18),
                     text_color="#ffffff").pack(pady=10)

        # Optionally show charts
        self.create_charts(scrollable_frame)

    def get_total_appointments_this_week(self):
        """Count how many appointments fall in the current calendar week."""
        today = datetime.date.today()
        start_week = today - datetime.timedelta(days=today.weekday())
        end_week = start_week + datetime.timedelta(days=6)
        get_cursor().execute('''
            SELECT COUNT(*)
            FROM appointments
            WHERE date BETWEEN ? AND ?
        ''', (start_week.strftime("%Y-%m-%d"), end_week.strftime("%Y-%m-%d")))
        return get_cursor().fetchone()[0]

    def get_returning_customers(self):
        """Customers who appear multiple times in the DB."""
        get_cursor().execute('''
            SELECT phone, COUNT(*) 
            FROM customers 
            GROUP BY phone 
            HAVING COUNT(*) > 1
        ''')
        returning = get_cursor().fetchall()
        return len(returning)

    def get_top_workers(self, top_n=3):
        get_cursor().execute('''
            SELECT worker, COUNT(*) as count
            FROM appointments
            GROUP BY worker
            ORDER BY count DESC
            LIMIT ?
        ''', (top_n,))
        return get_cursor().fetchall()

    def get_top_services(self, top_n=3):
        get_cursor().execute('''
            SELECT service, COUNT(*) as count
            FROM appointments
            GROUP BY service
            ORDER BY count DESC
            LIMIT ?
        ''', (top_n,))
        return get_cursor().fetchall()

    def get_appointments_per_service(self):
        get_cursor().execute('''
            SELECT service, COUNT(*)
            FROM appointments
            GROUP BY service
        ''')
        return get_cursor().fetchall()

    def get_appointments_per_worker(self):
        get_cursor().execute('''
            SELECT worker, COUNT(*)
            FROM appointments
            GROUP BY worker
        ''')
        return get_cursor().fetchall()

    def get_monthly_revenue(self):
        get_cursor().execute('''
            SELECT a.date, a.service
            FROM appointments a
        ''')
        appts = get_cursor().fetchall()
        revenue = {}
        for date_str, svc in appts:
            month = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%B %Y")
            svc_price = SERVICES.get(svc, {"price": 0})["price"]
            revenue[month] = revenue.get(month, 0) + svc_price
        # sort by actual date so months appear in chrono order
        sorted_rev = sorted(revenue.items(),
                            key=lambda x: datetime.datetime.strptime(x[0], "%B %Y"))
        return sorted_rev

    def create_charts(self, parent):
        """Show bar charts for 'Appointments per Service' and 'Appointments per Worker'."""
        services_data = self.get_appointments_per_service()
        if services_data:
            fig, ax = plt.subplots(figsize=(6,4))
            services = [row[0] for row in services_data]
            counts = [row[1] for row in services_data]
            ax.bar(services, counts, color='#4e9a06')
            ax.set_xlabel('Service', color='#ffffff')
            ax.set_ylabel('Number of Appointments', color='#ffffff')
            ax.set_title('Appointments per Service', color='#ffffff')
            ax.set_facecolor("#2e2e2e")
            fig.patch.set_facecolor('#2e2e2e')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color='#ffffff')
            plt.setp(ax.get_yticklabels(), color='#ffffff')
            plt.style.use('dark_background')
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=20, padx=20, fill="both", expand=True)

        workers_data = self.get_appointments_per_worker()
        if workers_data:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            workers = [row[0] for row in workers_data]
            wcounts = [row[1] for row in workers_data]
            ax2.bar(workers, wcounts, color='#ad7fa8')
            ax2.set_xlabel('Worker', color='#ffffff')
            ax2.set_ylabel('Number of Appointments', color='#ffffff')
            ax2.set_title('Appointments per Worker', color='#ffffff')
            ax2.set_facecolor("#2e2e2e")
            fig2.patch.set_facecolor('#2e2e2e')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', color='#ffffff')
            plt.setp(ax2.get_yticklabels(), color='#ffffff')
            plt.style.use('dark_background')
            canvas2 = FigureCanvasTkAgg(fig2, master=parent)
            canvas2.draw()
            canvas2.get_tk_widget().pack(pady=20, padx=20, fill="both", expand=True)

    def show_appointment_form(self):
        """A scrollable new-appointment form."""
        self.clear_main_frame()
        scrollable_frame = ctk.CTkScrollableFrame(self.main_frame, fg_color="#2e2e2e", scrollbar_button_color="#6e6e6e")
        scrollable_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(scrollable_frame, text="New Appointment",
                     font=("Helvetica", 24, "bold"), text_color="#ffffff").pack(pady=20)

        name_var = tk.StringVar()
        phone_var = tk.StringVar()
        service_var = tk.StringVar()
        date_var = tk.StringVar()
        time_var = tk.StringVar()
        worker_var = tk.StringVar()

        def submit_appointment():
            name = name_var.get().strip()
            phone = phone_var.get().strip()
            service = service_var.get()
            appt_date = date_var.get()
            appt_time = time_var.get()
            worker = worker_var.get()

            if not (name and phone and service and appt_date and appt_time and worker):
                messagebox.showwarning("Incomplete Data", "Please fill in all fields.")
                return

            # check date validity
            try:
                chosen_date = datetime.datetime.strptime(appt_date, "%Y-%m-%d").date()
                if chosen_date < datetime.date.today():
                    messagebox.showwarning("Invalid Date", "Cannot book in the past!")
                    return
            except ValueError:
                messagebox.showwarning("Invalid Date", "Please select a valid date.")
                return

            # Insert
            get_cursor().execute("INSERT INTO customers (name, phone) VALUES (?,?)", (name, phone))
            get_connection().commit()
            get_cursor().execute("SELECT id FROM customers WHERE name=? AND phone=?", (name, phone))
            customer = get_cursor().fetchone()
            if customer:
                customer_id = customer[0]
                get_cursor().execute('''
                    INSERT INTO appointments (customer_id, date, time, service, worker)
                    VALUES (?, ?, ?, ?, ?)
                ''', (customer_id, appt_date, appt_time, service, worker))
                get_connection().commit()
                messagebox.showinfo("Success", "Appointment added successfully!")
                self.clear_appointment_form()
            else:
                messagebox.showerror("Error", "Customer not found or insertion failed.")

        ctk.CTkLabel(scrollable_frame, text="Name:", font=("Helvetica", 14), text_color="#ffffff").pack()
        ctk.CTkEntry(scrollable_frame, textvariable=name_var, width=200).pack(pady=5)

        ctk.CTkLabel(scrollable_frame, text="Phone:", font=("Helvetica", 14), text_color="#ffffff").pack()
        ctk.CTkEntry(scrollable_frame, textvariable=phone_var, width=200).pack(pady=5)

        ctk.CTkLabel(scrollable_frame, text="Service:", font=("Helvetica", 14), text_color="#ffffff").pack()
        service_menu = ctk.CTkOptionMenu(scrollable_frame, values=list(SERVICES.keys()), variable=service_var)
        service_menu.pack(pady=5)

        ctk.CTkLabel(scrollable_frame, text="Date:", font=("Helvetica", 14), text_color="#ffffff").pack()
        cal = Calendar(scrollable_frame, selectmode='day', date_pattern='yyyy-mm-dd')
        cal.pack(pady=5)

        def select_date():
            date_var.set(cal.get_date())

        ctk.CTkButton(scrollable_frame, text="Select Date", command=select_date).pack(pady=5)

        ctk.CTkLabel(scrollable_frame, text="Time (HH:MM):", font=("Helvetica", 14), text_color="#ffffff").pack()
        ctk.CTkEntry(scrollable_frame, textvariable=time_var, width=200).pack(pady=5)

        ctk.CTkLabel(scrollable_frame, text="Worker:", font=("Helvetica", 14), text_color="#ffffff").pack()
        worker_menu = ctk.CTkOptionMenu(scrollable_frame, values=WORKERS, variable=worker_var)
        worker_menu.pack(pady=5)

        ctk.CTkButton(scrollable_frame, text="Submit", command=submit_appointment).pack(pady=10)

    def clear_appointment_form(self):
        self.show_appointment_form()

    def view_appointments(self):
        """Show all appointments from DB."""
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="View Appointments",
                     font=("Helvetica", 24, "bold"),
                     text_color="#ffffff").pack(pady=20)

        scrollable_frame = ctk.CTkScrollableFrame(self.main_frame,
                                                  fg_color="#2e2e2e",
                                                  scrollbar_button_color="#6e6e6e")
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)

        get_cursor().execute('''
            SELECT c.name, c.phone, a.date, a.time, a.service, a.worker
            FROM appointments a
            JOIN customers c ON a.customer_id = c.id
            ORDER BY a.date, a.time
        ''')
        appointments = get_cursor().fetchall()

        for appt in appointments:
            info = (f"Name: {appt[0]}, Phone: {appt[1]}, "
                    f"Date: {appt[2]}, Time: {appt[3]}, "
                    f"Service: {appt[4]}, Worker: {appt[5]}")
            ctk.CTkLabel(scrollable_frame, text=info,
                         text_color="#ffffff",
                         anchor="w", justify="left").pack(pady=2, padx=10, fill="x")

    def search_appointments(self):
        """Provide a search box."""
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="Search Appointments",
                     font=("Helvetica", 24, "bold"),
                     text_color="#ffffff").pack(pady=20)

        search_var = tk.StringVar()

        def do_search():
            query = search_var.get()
            self.execute_search(query)

        ctk.CTkEntry(self.main_frame, textvariable=search_var, width=200).pack(pady=5)
        ctk.CTkButton(self.main_frame, text="Search", command=do_search).pack(pady=10)

    def execute_search(self, query):
        """Run the DB query to find matching appointments."""
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame,
                     text=f"Search Results for '{query}'",
                     font=("Helvetica", 24, "bold"),
                     text_color="#ffffff").pack(pady=20)

        scrollable_frame = ctk.CTkScrollableFrame(self.main_frame,
                                                  fg_color="#2e2e2e",
                                                  scrollbar_button_color="#6e6e6e")
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=20)

        get_cursor().execute('''
            SELECT c.name, c.phone, a.date, a.time, a.service, a.worker
            FROM appointments a
            JOIN customers c ON a.customer_id = c.id
            WHERE c.name LIKE ? OR c.phone LIKE ? OR a.service LIKE ? OR a.worker LIKE ?
            ORDER BY a.date, a.time
        ''', (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%"))
        appointments = get_cursor().fetchall()

        if not appointments:
            ctk.CTkLabel(scrollable_frame, text="No results found.",
                         text_color="#ffffff").pack(pady=10)
        else:
            for appt in appointments:
                info = (f"Name: {appt[0]}, Phone: {appt[1]}, "
                        f"Date: {appt[2]}, Time: {appt[3]}, "
                        f"Service: {appt[4]}, Worker: {appt[5]}")
                ctk.CTkLabel(scrollable_frame, text=info,
                             text_color="#ffffff", anchor="w",
                             justify="left").pack(pady=2, padx=10, fill="x")

    def ai_conversation(self):
        """Simple AI chat interface."""
        self.clear_main_frame()
        ctk.CTkLabel(self.main_frame, text="AI Conversation",
                     font=("Helvetica", 24, "bold"),
                     text_color="#ffffff").pack(pady=20)

        user_input_var = tk.StringVar()

        def send_message():
            user_input = user_input_var.get().strip()
            if user_input:
                response = get_bot_response(user_input)
                log_conversation(user_input, response)
                ctk.CTkLabel(self.main_frame, text=f"User: {user_input}",
                             text_color="#ffffff", anchor="w",
                             justify="left").pack(pady=2, padx=10, fill="x")
                ctk.CTkLabel(self.main_frame, text=f"AI: {response}",
                             text_color="#ffffff", anchor="w",
                             justify="left").pack(pady=2, padx=10, fill="x")
                user_input_var.set("")

        ctk.CTkEntry(self.main_frame, textvariable=user_input_var, width=300).pack(pady=5)
        ctk.CTkButton(self.main_frame, text="Send", command=send_message).pack(pady=10)

    def run_and_compare_schedules(self):
        """
        Pull all appointments from DB, run Greedy & GA,
        then show summary in the console.
        """
        get_cursor().execute('''
            SELECT c.name, c.phone, a.date, a.time, a.service, a.worker
            FROM appointments a
            JOIN customers c ON a.customer_id = c.id
        ''')
        data = get_cursor().fetchall()
        if not data:
            print("No appointments in the dataset.")
            return

        appointments = []
        for row in data:
            appt = {
                'name': row[0],
                'phone': row[1],
                'date': row[2],
                'time': row[3],
                'service': row[4],
                'worker': row[5]
            }
            appointments.append(appt)

        print("=== Running Greedy Scheduling ===")
        greedy_sched = run_greedy_schedule(appointments)

        print("=== Running Genetic Algorithm ===")
        best_sched = run_genetic_algorithm(appointments, generations=300,
                                           population_size=100,
                                           mutation_rate=0.4,
                                           attempts=1)

        # Summaries
        g_conf = count_conflicts(greedy_sched)
        ga_conf = count_conflicts(best_sched)
        g_idle = calculate_total_idle_time(greedy_sched)
        ga_idle = calculate_total_idle_time(best_sched)
        g_rev = calculate_total_revenue(greedy_sched)
        ga_rev = calculate_total_revenue(best_sched)

        print("\n=== SCHEDULING COMPARISON SUMMARY ===")
        print(f"Greedy => Conflicts: {g_conf}, Idle: {g_idle:.2f}, Revenue: ${g_rev}")
        print(f"GA     => Conflicts: {ga_conf}, Idle: {ga_idle:.2f}, Revenue: ${ga_rev}")

        # Compare
        if ga_conf < g_conf:
            print(f"  - GA improved conflicts from {g_conf} to {ga_conf}!")
        elif ga_conf > g_conf:
            print(f"  - GA has more conflicts: {ga_conf} vs {g_conf}.")
        else:
            print(f"  - Both have {ga_conf} conflicts.")

        if ga_idle < g_idle:
            print(f"  - GA idle time improved from {g_idle:.2f} to {ga_idle:.2f}.")
        elif ga_idle > g_idle:
            print(f"  - GA idle time is higher: {ga_idle:.2f} vs {g_idle:.2f}.")
        else:
            print(f"  - Both have {g_idle:.2f} idle time.")

        if ga_rev > g_rev:
            print(f"  - GA revenue increased from ${g_rev} to ${ga_rev}.")
        elif ga_rev < g_rev:
            print(f"  - GA revenue is lower: ${ga_rev} vs ${g_rev}.")
        else:
            print(f"  - Both have ${g_rev} revenue.")

        if ga_conf < g_conf:
            print("\nConclusion: GA found fewer conflicts than Greedy.")
        else:
            print("\nConclusion: GA did not reduce conflicts or matched them.")

        print_worker_comparison(appointments, greedy_sched, best_sched)

        print("\n=== Greedy Schedule (sorted) ===")
        for ga in sort_schedule_by_date_time(greedy_sched):
            print(f"- {ga['date']} {ga['time']}: {ga['service']} | Worker: {ga['worker']} | Customer: {ga['name']}")

        print("\n=== GA Schedule (sorted) ===")
        for bs in sort_schedule_by_date_time(best_sched):
            print(f"- {bs['date']} {bs['time']}: {bs['service']} | Worker: {bs['worker']} | Customer: {bs['name']}")

    def on_closing(self):
        """Ask user to confirm, then close DB connection and quit."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.running = False
            get_connection().close()
            self.master.quit()
            self.master.destroy()

    def run(self):
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.mainloop()

if __name__ == "__main__":
    # Load data from file if DB is empty
    load_data_from_file()
    root = ctk.CTk()
    app = SalonSchedulerApp(root)
    app.run()
