import sqlite3

def view_database():
    conn = sqlite3.connect('salon_schedule.db')
    cursor = conn.cursor()

    # Fetch and display customers
    print("=== Customers ===")
    cursor.execute('SELECT * FROM customers')
    customers = cursor.fetchall()
    for customer in customers:
        print(f"ID: {customer[0]}, Name: {customer[1]}, Phone: {customer[2]}")
    print()

    # Fetch and display appointments
    print("=== Appointments ===")
    cursor.execute('SELECT * FROM appointments')
    appointments = cursor.fetchall()
    for appointment in appointments:
        print(f"ID: {appointment[0]}, Customer ID: {appointment[1]}, Date: {appointment[2]}, Time: {appointment[3]}, Service: {appointment[4]}, Worker: {appointment[5]}")
    print()

    # Fetch and display appointments with customer details
    print("=== Appointments with Customer Details ===")
    cursor.execute('''
        SELECT a.id, c.name, c.phone, a.date, a.time, a.service, a.worker
        FROM appointments a
        JOIN customers c ON a.customer_id = c.id
        ORDER BY a.date, a.time
    ''')
    appointments_with_customers = cursor.fetchall()
    for appt in appointments_with_customers:
        print(f"Appointment ID: {appt[0]}, Customer Name: {appt[1]}, Phone: {appt[2]}, Date: {appt[3]}, Time: {appt[4]}, Service: {appt[5]}, Worker: {appt[6]}")

    cursor.close()
    conn.close()

if __name__ == '__main__':
    view_database()
