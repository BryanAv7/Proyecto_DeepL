import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
import { catchError, Observable, throwError } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class PredictService {
  private apiUrl = 'http://127.0.0.1:8000/api/predict-product';

  constructor(private http: HttpClient) { }

  predictImage(imageFile: File): Observable<{ result: string }> {
    const formData = new FormData();
    formData.append('file', imageFile);

    return this.http.post<any>(this.apiUrl, formData).pipe(
      catchError(this.handleError)
    );
  }

  private handleError(error: HttpErrorResponse) {
    console.error('Error en la petición:', error);
    return throwError(() => new Error('Error en el servidor'));
  }
}
