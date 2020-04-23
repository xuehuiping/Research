process.py 将原始文件train、valid、test转化为jsonl文件。
每行文件的格式是：src、knowledge、tgt

原始数据：
a person_2054 is watching a video on a laptop __eou__ the person_2054 begins eating a cookie , then places the laptop on a shelf __eou__ the man is sitting at the table looking at the laptop __eou__ he stands up puts the laptop up	what is happening in the video ?	the man is sitting at the table looking at the laptop . he stands up puts the laptop up .

转化之后：
{
	"src": [
		[2054, 2003, 6230, 1999, 1996, 2678, 1029, 2]
	],
	"knowledge": [
		[1037, 2711, 2003, 3666, 1037, 2678, 2006, 1037, 12191, 2],
		[1996, 2711, 4269, 5983, 1037, 17387, 1010, 2059, 3182, 1996, 12191, 2006, 1037, 11142, 2],
		[1996, 2158, 2003, 3564, 2012, 1996, 2795, 2559, 2012, 1996, 12191, 2],
		[2002, 4832, 2039, 8509, 1996, 12191, 2039, 2]
	],
	"tgt": [1, 1996, 2158, 2003, 3564, 2012, 1996, 2795, 2559, 2012, 1996, 12191, 1012, 2002, 4832, 2039, 8509, 1996, 12191, 2039, 1012, 2]
}