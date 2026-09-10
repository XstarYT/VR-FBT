class Const:
	BasePath = '/tracking/trackers/'

	class Type:
		Position = 'position'
		Rotation = 'rotation'

class Phrase:
	@staticmethod
	def list(OSCList):
		return ( OSCList[0]+OSCList[1]+'/'+OSCList[2], OSCList[3] )

	@staticmethod
	def dict(OSCDict):
		return (OSCDict['BasePath']+OSCDict['TrackerID']+'/'+OSCDict['Type'], OSCDict['Data'])

	@staticmethod
	def str(OSCStr): # '{BasePath}{TrackerID}/{Type}|{Data}'
		import ast
		msg = OSCStr.split('|', 1)
		if len(msg) != 2:
			raise ValueError('OSC string must contain a | separator')
		return (msg[0], ast.literal_eval(msg[1]))

	@staticmethod
	def direct(BasePath, TrackerID, Type, Data):
		return (BasePath+TrackerID+'/'+Type, Data)

	@staticmethod
	def debug(Name, Data):
		return ('Debug/'+Name, Data)

class Server:
	def __init__(self, ip, port=9000):
		import socket
		from pythonosc.udp_client import SimpleUDPClient
		# VRChat's default OSC listener binds IPv4 (0.0.0.0:9000). On
		# Windows, "localhost" commonly resolves to IPv6 ::1 first, which
		# makes python-osc silently send to a socket VRChat is not listening on.
		self.target = (socket.gethostbyname(ip), int(port))
		self.client = SimpleUDPClient(self.target[0], self.target[1], family=socket.AF_INET)

	def Send(self, msg):
		self.client.send_message(msg[0], msg[1])

	def close(self):
		self.client._sock.close()


# Backward-compatible alias for the original misspelling.
Pharse = Phrase

