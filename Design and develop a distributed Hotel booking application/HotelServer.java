
import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.HashMap;
import java.util.Map;

public class HotelServer extends UnicastRemoteObject implements Hotel {
    private final Map<String, Boolean> roomBookings;

    protected HotelServer() throws RemoteException {
        super();
        roomBookings = new HashMap<>();
    }

    @Override
    public String bookRoom(String guestName) throws RemoteException {
        if (roomBookings.containsKey(guestName) && roomBookings.get(guestName)) {
            return "Booking failed: Room already booked for " + guestName;
        }
        roomBookings.put(guestName, true);
        return "Room successfully booked for " + guestName;
    }

    @Override
    public String cancelBooking(String guestName) throws RemoteException {
        if (!roomBookings.containsKey(guestName) || !roomBookings.get(guestName)) {
            return "Cancellation failed: No booking found for " + guestName;
        }
        roomBookings.put(guestName, false);
        return "Booking cancelled for " + guestName;
    }

    public static void main(String[] args) {
        try {
            LocateRegistry.createRegistry(1099); // Start RMI registry on port 1099
            HotelServer server = new HotelServer();
            Registry registry = LocateRegistry.getRegistry();
            registry.bind("HotelService", server);
            System.out.println("Hotel Server is ready.");
        } catch (Exception e) {
            System.err.println("Server exception: " + e.toString());
            e.printStackTrace();
        }
    }
}
