import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class SaludoService {
  private apiUrl = 'http://localhost:8000/api/saludo';

  constructor(private http: HttpClient) {}

  obtenerSaludo(): Observable<any> {
    return this.http.get<any>(this.apiUrl);
  }
}
