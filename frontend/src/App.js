import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Alert, AlertDescription } from './components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { ScrollArea } from './components/ui/scroll-area';
import { Separator } from './components/ui/separator';
import { MessageCircle, Mic, Volume2, Brain, Shield, Database, Search, AlertTriangle } from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

function App() {
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [distressAlert, setDistressAlert] = useState(null);
  const [ragQuery, setRagQuery] = useState('');
  const [ragResults, setRagResults] = useState([]);
  const [sessionSummary, setSessionSummary] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchSystemStatus();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const fetchSystemStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/`);
      setSystemStatus(response.data);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  };

  const sendMessage = async () => {
    if (!currentMessage.trim() || isLoading) return;

    const userMessage = { role: 'user', content: currentMessage, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setDistressAlert(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: currentMessage,
        session_id: sessionId,
        include_context: true
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date(),
        context_used: response.data.context_used || []
      };

      setMessages(prev => [...prev, assistantMessage]);
      setSessionId(response.data.session_id);

      // Handle distress detection
      if (response.data.distress_detected) {
        setDistressAlert({
          detected: true,
          helpline: response.data.helpline_info
        });
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your message. Please try again.',
        timestamp: new Date(),
        error: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setCurrentMessage('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const searchKnowledgeBase = async () => {
    if (!ragQuery.trim()) return;

    try {
      const response = await axios.post(`${API_BASE_URL}/api/rag/search`, {
        query: ragQuery,
        top_k: 3
      });
      setRagResults(response.data.results || []);
    } catch (error) {
      console.error('Failed to search knowledge base:', error);
    }
  };

  const fetchSessionSummary = async () => {
    if (!sessionId) return;

    try {
      const response = await axios.get(`${API_BASE_URL}/api/sessions/${sessionId}/summary`);
      setSessionSummary(response.data);
    } catch (error) {
      console.error('Failed to fetch session summary:', error);
    }
  };

  const testTTS = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/synthesize`, {
        text: "Hello! This is a test of the text-to-speech system.",
        voice: "default"
      });
      
      if (response.data.audio_path) {
        // In a real implementation, you would play the audio file
        alert(`Audio generated successfully! File size: ${response.data.size} bytes`);
      }
    } catch (error) {
      console.error('TTS test failed:', error);
    }
  };

  const clearMessages = () => {
    setMessages([]);
    setSessionId(null);
    setDistressAlert(null);
    setSessionSummary(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto p-6 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-full flex items-center justify-center">
              <span className="text-white text-xl font-bold">🌵</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Sahara AI Assistant</h1>
              <p className="text-gray-600">Empathetic AI with multimodal emotion detection</p>
            </div>
          </div>
          
          {/* System Status */}
          {systemStatus && (
            <div className="flex gap-2 mb-4">
              <Badge variant="secondary" className="bg-green-100 text-green-800">
                {systemStatus.status}
              </Badge>
              <Badge variant="outline">
                Version {systemStatus.version}
              </Badge>
              {systemStatus.demo_mode?.gemini && (
                <Badge variant="outline" className="bg-yellow-100 text-yellow-800">
                  Gemini Demo Mode
                </Badge>
              )}
              {systemStatus.demo_mode?.riva && (
                <Badge variant="outline" className="bg-blue-100 text-blue-800">
                  Riva Demo Mode
                </Badge>
              )}
            </div>
          )}
        </div>

        {/* Distress Alert */}
        {distressAlert?.detected && (
          <Alert className="mb-6 border-red-200 bg-red-50">
            <AlertTriangle className="h-4 w-4 text-red-600" />
            <AlertDescription className="text-red-800">
              <div className="font-semibold mb-2">Crisis Support Available</div>
              {distressAlert.helpline && (
                <div>
                  <p><strong>{distressAlert.helpline.name}</strong></p>
                  {distressAlert.helpline.phone && (
                    <p>Phone: <a href={`tel:${distressAlert.helpline.phone}`} className="underline font-semibold">{distressAlert.helpline.phone}</a></p>
                  )}
                  {distressAlert.helpline.text && (
                    <p>Text: {distressAlert.helpline.text}</p>
                  )}
                </div>
              )}
            </AlertDescription>
          </Alert>
        )}

        <Tabs defaultValue="chat" className="w-full">
          <TabsList className="grid w-full grid-cols-4 lg:w-[400px]">
            <TabsTrigger value="chat" className="flex items-center gap-2">
              <MessageCircle className="w-4 h-4" />
              Chat
            </TabsTrigger>
            <TabsTrigger value="knowledge" className="flex items-center gap-2">
              <Search className="w-4 h-4" />
              Search
            </TabsTrigger>
            <TabsTrigger value="features" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Features
            </TabsTrigger>
            <TabsTrigger value="memory" className="flex items-center gap-2">
              <Database className="w-4 h-4" />
              Memory
            </TabsTrigger>
          </TabsList>

          {/* Chat Tab */}
          <TabsContent value="chat" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <MessageCircle className="w-5 h-5" />
                    Conversation
                  </CardTitle>
                  <CardDescription>
                    Chat with Sahara - your empathetic AI assistant with emotion awareness
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96 w-full rounded-md border p-4 mb-4">
                    {messages.length === 0 ? (
                      <div className="text-center text-gray-500 py-8">
                        <MessageCircle className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                        <p>Start a conversation with Sahara!</p>
                        <p className="text-sm mt-2">Try asking about emotions, mental health, or general topics.</p>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {messages.map((message, index) => (
                          <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[80%] rounded-lg px-4 py-2 ${
                              message.role === 'user' 
                                ? 'bg-blue-600 text-white' 
                                : message.error 
                                  ? 'bg-red-100 text-red-800 border border-red-200'
                                  : 'bg-gray-100 text-gray-900'
                            }`}>
                              <p className="whitespace-pre-wrap">{message.content}</p>
                              <div className="flex items-center gap-2 mt-2 text-xs opacity-70">
                                <span>{message.timestamp.toLocaleTimeString()}</span>
                                {message.context_used && message.context_used.length > 0 && (
                                  <Badge variant="outline" className="text-xs">
                                    Used KB: {message.context_used.length} sources
                                  </Badge>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                        <div ref={messagesEndRef} />
                      </div>
                    )}
                  </ScrollArea>
                  
                  <div className="flex gap-2">
                    <Textarea
                      placeholder="Type your message here... (Press Enter to send, Shift+Enter for new line)"
                      value={currentMessage}
                      onChange={(e) => setCurrentMessage(e.target.value)}
                      onKeyPress={handleKeyPress}
                      className="min-h-[60px] resize-none"
                      disabled={isLoading}
                    />
                    <div className="flex flex-col gap-2">
                      <Button 
                        onClick={sendMessage} 
                        disabled={!currentMessage.trim() || isLoading}
                        className="px-6"
                      >
                        {isLoading ? 'Sending...' : 'Send'}
                      </Button>
                      <Button variant="outline" onClick={clearMessages} size="sm">
                        Clear
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Chat Info Panel */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Session Info</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Session ID</p>
                    <p className="text-xs text-gray-500 font-mono">{sessionId || 'Not started'}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-600">Messages</p>
                    <p className="text-sm">{messages.length}</p>
                  </div>
                  <Separator />
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium flex items-center gap-2">
                      <Shield className="w-4 h-4" />
                      Safety Features
                    </h4>
                    <ul className="text-xs text-gray-600 space-y-1">
                      <li>• Crisis detection active</li>
                      <li>• Helpline resources available</li>
                      <li>• Privacy-first design</li>
                      <li>• Local data storage</li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Knowledge Search Tab */}
          <TabsContent value="knowledge" className="mt-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Search className="w-5 h-5" />
                  Knowledge Base Search
                </CardTitle>
                <CardDescription>
                  Search the knowledge base using semantic similarity and keyword matching
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder="Search for information..."
                    value={ragQuery}
                    onChange={(e) => setRagQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && searchKnowledgeBase()}
                  />
                  <Button onClick={searchKnowledgeBase}>Search</Button>
                </div>
                
                {ragResults.length > 0 && (
                  <div className="space-y-3">
                    <h4 className="font-medium">Search Results:</h4>
                    {ragResults.map((result, index) => (
                      <Card key={index} className="border-l-4 border-l-blue-500">
                        <CardContent className="p-4">
                          <div className="flex justify-between items-start mb-2">
                            <Badge variant="outline">{result.source}</Badge>
                            <div className="text-xs text-gray-500">
                              Score: {result.semantic_score?.toFixed(3) || 'N/A'}
                            </div>
                          </div>
                          <p className="text-sm">{result.content}</p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Features Tab */}
          <TabsContent value="features" className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Mic className="w-5 h-5" />
                    Speech Processing
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 mb-4">
                    Speech-to-text and text-to-speech capabilities using NVIDIA Riva
                  </p>
                  <Button onClick={testTTS} variant="outline" className="w-full">
                    <Volume2 className="w-4 h-4 mr-2" />
                    Test Text-to-Speech
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5" />
                    Emotion Detection
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 mb-4">
                    Multimodal emotion analysis from voice, video, and text
                  </p>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span>Audio Emotion:</span>
                      <Badge variant="secondary">Demo Mode</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Video Emotion:</span>
                      <Badge variant="secondary">Demo Mode</Badge>
                    </div>
                    <div className="flex justify-between">
                      <span>Text Analysis:</span>
                      <Badge variant="default">Active</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="w-5 h-5" />
                    Safety Monitoring
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-gray-600 mb-4">
                    Crisis detection and helpline resource provision
                  </p>
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span>Distress monitoring active</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span>Helpline resources ready</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span>Privacy-preserving design</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Memory Tab */}
          <TabsContent value="memory" className="mt-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="w-5 h-5" />
                    Session Summary
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Button 
                    onClick={fetchSessionSummary} 
                    disabled={!sessionId}
                    className="w-full mb-4"
                  >
                    Generate Session Summary
                  </Button>
                  
                  {sessionSummary && (
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm font-medium">Turn Count</p>
                        <p className="text-lg">{sessionSummary.turn_count}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Summary</p>
                        <p className="text-sm text-gray-600">{sessionSummary.summary}</p>
                      </div>
                      {sessionSummary.latest_timestamp && (
                        <div>
                          <p className="text-sm font-medium">Last Activity</p>
                          <p className="text-xs text-gray-500">{new Date(sessionSummary.latest_timestamp).toLocaleString()}</p>
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Memory Management</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4 text-sm">
                    <div>
                      <h4 className="font-medium mb-2">Short-term Memory</h4>
                      <p className="text-gray-600">Maintains context for recent conversation turns</p>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Long-term Memory</h4>
                      <p className="text-gray-600">Stores conversation summaries and patterns</p>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Context Integration</h4>
                      <p className="text-gray-600">Uses memory to provide personalized responses</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default App;