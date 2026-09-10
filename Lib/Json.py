import json

class File:
	@staticmethod
	def load(path):
		with open(path, 'r', encoding='utf-8') as file:
			out = json.load(file)
		return out

	@staticmethod	
	def save(path, data, indent=4):
		with open(path, 'w', encoding='utf-8') as file:
			json.dump(data, file, indent=indent)

class String:
	@staticmethod
	def load(Str):
		return json.loads(Str)

	@staticmethod
	def toStr(data, indent=2):
		return json.dumps(data, indent=indent)
